import datetime
from collections.abc import Callable, Sequence, Mapping
import functools
import types
import abc
import threading
import pandas as pd

from concordia.agents import entity_agent_with_logging
from concordia.components.agent import memory_component
from concordia.associative_memory import associative_memory
from concordia.associative_memory import formative_memories
from concordia.clocks import game_clock
from concordia.document import interactive_document
from concordia.components import agent as agent_components
from concordia.language_model import language_model
from concordia.memory_bank import legacy_associative_memory
from concordia.utils import measurements as measurements_lib
from concordia.components.agent import question_of_recent_memories
from concordia.utils import concurrency
from concordia.typing import entity_component
from concordia.typing import entity as entity_lib
from concordia.typing import logging
from concordia.components.agent import action_spec_ignored
from concordia.utils import helper_functions
from typing_extensions import override
from typing import Final, Any
from concordia.typing import memory as memory_lib
from absl import logging as absl_logging

_ASSOCIATIVE_RETRIEVAL = legacy_associative_memory.RetrieveAssociative()


def _get_all_memories(
    memory_component_: agent_components.memory_component.MemoryComponent,
    add_time: bool = True,
    sort_by_time: bool = True,
    constant_score: float = 0.0,
) -> Sequence[memory_lib.MemoryResult]:
    """Returns all memories in the memory bank.

    Args:
      memory_component_: The memory component to retrieve memories from.
      add_time: whether to add time
      sort_by_time: whether to sort by time
      constant_score: assign this score value to each memory
    """
    texts = memory_component_.get_all_memories_as_text(
        add_time=add_time, sort_by_time=sort_by_time
    )
    return [
        memory_lib.MemoryResult(text=t, score=constant_score) for t in texts
    ]


def _get_earliest_timepoint(
    memory_component_: agent_components.memory_component.MemoryComponent,
) -> datetime.datetime:
    """Returns all memories in the memory bank.

    Args:
      memory_component_: The memory component to retrieve memories from.
    """
    memories_data_frame = memory_component_.get_raw_memory()
    if not memories_data_frame.empty:
        sorted_memories_data_frame = memories_data_frame.sort_values(
            "time", ascending=True
        )
        return sorted_memories_data_frame["time"][0]
    else:
        absl_logging.warn("No memories found in memory bank.")
        return datetime.datetime.now()


class Situations(action_spec_ignored.ActionSpecIgnored):
    """
    Component to respresent the situations our agent finds themselves
    in.
    """

    def __init__(
        self,
        model: language_model.LanguageModel,
        clock,
        memory_component_name: str = (
            memory_component.DEFAULT_MEMORY_COMPONENT_NAME
        ),
        pre_act_key: str = "The current situation",
        logging_channel: logging.LoggingChannel = logging.NoOpLoggingChannel,
    ):
        """Initialize a component to consider the current situation.

        Args:
            model: The language model to use.
            clock_now: Function that returns the current time.
            memory_component_name: The name of the memory component from which
                to retrieve related memories.
            pre_act_key: Prefix to add to the output of the component when
                called in `pre_act`.
            logging_channel: The channel to log debug information to.
        """
        super().__init__(pre_act_key)
        self._model = model
        self._clock = clock
        self._memory_component_name = memory_component_name
        self._logging_channel = logging_channel
        self._previous_time = None
        self._current_situation = None
        self._log = {"Key": pre_act_key}
        self._situations =  pd.DataFrame(
            columns=['description', 'interval', 'people', 'observations']
        )

    def _make_pre_act_value(self) -> str:
        """Returns a representation of the current situation to pre act."""
        agent_name = self.get_entity().name
        current_time = self._clock.now()
        memory = self.get_entity().get_component(
            self._memory_component_name, type_=memory_component.MemoryComponent
        )
        if self._current_situation is None:
            self._previous_time = _get_earliest_timepoint(memory)
            mems = "\n".join([mem.text for mem in _get_all_memories(memory)])
            current_situation_prompt = (
                interactive_document.InteractiveDocument(self._model)
            )
            current_situation_prompt.statement(
                f"Observations of {agent_name} so far:\n{mems}"
            )
            current_time_str = self._clock.current_time_interval_str()
            current_situation_prompt.statement(
                f"Current date and time: {current_time_str}"
            )
            question = (
                "Thoroughly analyse the above to determine the current "
                f"situation that {agent_name} is in."
            )
            self._current_situation = current_situation_prompt.open_question(
                question=question,
                max_tokens=1000,
                terminators=(),
                question_label="Exercise",
            ).rstrip("\n")
            self._logging_channel(
                {
                    "Key": self.get_pre_act_key(),
                    "Current Situation": self._current_situation,
                    "Chain of thought": current_situation_prompt.view()
                    .text()
                    .splitlines(),
                }
            )
            self._previous_time = current_time
            return self._current_situation
        interval_scorer = legacy_associative_memory.RetrieveTimeInterval(
            time_from=self._previous_time,# - self._clock.get_step_size(),
            time_until=current_time,
            add_time=True,
        )
        mems = [
            mem.text for mem in memory.retrieve(scoring_fn=interval_scorer)
        ]
        recent_obs = "\n".join(mems)
        change_situation_prompt = interactive_document.InteractiveDocument(
            self._model
        )
        change_situation_prompt.statement(
            f"Previous situation description:\n{self._current_situation}"
        )
        change_situation_prompt.statement(
            f"Recent observations of {agent_name}:\n{recent_obs}"
        )
        current_time_str = self._clock.current_time_interval_str()
        change_situation_prompt.statement(
            f"Current date and time: {current_time_str}"
        )
        question = (
            f"Thoroughly analyse the above recent observations, paying close "
            f"attention to the times. Has the current situation {agent_name} "
            "is in changed?"
        )
        situation_change = change_situation_prompt.yes_no_question(question)
        if situation_change:
            contents = {
                'text': self._current_situation,
                'interval': time_from=self_prev_situation_start_time,
                'tags': tuple(tags),
                'importance': importance,
            }
            hashed_contents = hash(tuple(contents.values()))
            derived = {'embedding': self._embedder(text)}
            new_df = pd.Series(contents | derived).to_frame().T.infer_objects()

            with self._memory_bank_lock:
            if hashed_contents in self._stored_hashes:
                return
            self._memory_bank = pd.concat(
                [self._memory_bank, new_df], ignore_index=True
            )
            self._stored_hashes.add(hashed_contents)
            current_situation_prompt = (
                interactive_document.InteractiveDocument(self._model)
            )
            current_situation_prompt.statement(
                f"Recent observations of {agent_name}:\n{recent_obs}"
            )
            current_time_str = self._clock.current_time_interval_str()
            current_situation_prompt.statement(
                f"Current date and time: {current_time_str}"
            )
            question = (
                "Thoroughly analyse the above to determine the current "
                f"situation that {agent_name} is in. Pay close attention to "
                "the date and times of the observations."
            )
            self._current_situation = current_situation_prompt.open_question(
                question=question,
                max_tokens=1000,
                terminators=(),
                question_label="Exercise",
            ).rstrip("\n")
            self._log.update(
                {
                    "Current Situation CoT": current_situation_prompt.view()
                    .text()
                    .splitlines()
                }
            )
        self._log.update(
            {
                "Current Situation": self._current_situation,
                "Change Situation CoT": change_situation_prompt.view()
                .text()
                .splitlines(),
            }
        )
        self._logging_channel(self._log)
        self._log = {"Key": self.get_pre_act_key()}
        self._previous_time = current_time
        return self._current_situation
