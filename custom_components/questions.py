import datetime
from collections.abc import Callable, Sequence, Mapping
import functools
import types
import abc
import threading
import re

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

class Questions(action_spec_ignored.ActionSpecIgnored):
    """
    Component for gathering information from observations using LLM generated
    questions and answers.
    """

    def __init__(
        self,
        model: language_model.LanguageModel,
        observation: entity_component.ComponentName,
        pre_act_key: str,
        memory_component_name: str = (
            memory_component.DEFAULT_MEMORY_COMPONENT_NAME
        ),
        num_memories_to_retrieve: int = 20,
        clock=None,
        logging_channel: logging.LoggingChannel = logging.NoOpLoggingChannel,
    ):
        """
        Args:
            model: a language model
                observation_component_name: The name of the observation
                component from which to retrieve obervations.
            observation: name of the agent's observation component.
            pre_act_key: Prefix to add to the output of the component when
                called in `pre_act`.
            clock: time callback to use for the state.
            logging_channel: channel to use for debug logging.
        """
        super().__init__(pre_act_key)
        self._model = model
        self._observation = observation
        self._memory_component_name = memory_component_name
        self._num_memories_to_retrieve = num_memories_to_retrieve
        self._clock = clock
        self._logging_channel = logging_channel

        self._lock = threading.Lock()
        self._log = {"Key": pre_act_key}
        self._pre_act_key = pre_act_key
        # self._prev_answers = ''

    def _make_pre_act_value(self):
        return ""

    def _query_memory(self, query):
        memory = self.get_entity().get_component(
            self._memory_component_name, type_=memory_component.MemoryComponent
        )
        mems = "\n".join([mem.text for mem in _get_all_memories(memory)])
        self._log.update({"All Memories": mems})
        mems = [
            mem.text
            for mem in memory.retrieve(
                query=query,
                scoring_fn=_ASSOCIATIVE_RETRIEVAL,
                limit=self._num_memories_to_retrieve,
            )
        ]
        return mems

    def generate_information(
        self,
        call_to_action: str,
    ) -> str:
        agent_name = self.get_entity().name
        observation = (
            f"{agent_name} has just observed the following:\n"
            f"{self.get_named_component_pre_act_value(self._observation)}"
        )
        prompt = interactive_document.InteractiveDocument(self._model)
        prompt.statement(observation)
        if self._clock is not None:
            prompt.statement(
                f"Current time: {self._clock.current_time_interval_str()}.\n"
            )
        prompt.statement(f"Call to action: {call_to_action}")
        prompt.statement("\n")
        question = (
            f"Later you will have to give advice to {agent_name} on how to "
            "respond to the call to action. Write a list of questions you "
            "would need answers to in order to provide advice."
        )
        questions = prompt.open_question(
            question,
            max_tokens=2000,
            terminators=(),
            question_label="Exercise",
        ).rstrip("\n")
        CoT_break = len(prompt.view().text().splitlines())
        self._log.update(
            {
                "Questions CoT": prompt.view().text().splitlines(),
                "Questions": questions,
            }
        )
        prompt.statement("\n\n")
        question = (
            f"You have access to a database of {agent_name}'s memories. What "
            "keywords would you use to search this database in order to help "
            "answer the questions posed? "
            "Write a short list of keywords to search the database of "
            "memories. Please output only the list of keywords separated by "
            "commas. For example, if the keywords chosen were 'money', 'goal' "
            "and 'relationships' the output would be "
            "'money,goal,relationships'."
        )
        keywords = prompt.open_question(
            question,
            question_label="Exercise",
        ).split(",")
        keywords = [query.strip() for query in keywords]
        self._log.update(
            {
                "Keywords CoT": prompt.view().text().splitlines()[CoT_break:],
                "Memory Keywords": keywords,
            }
        )
        CoT_break = len(prompt.view().text().splitlines())
        relevant_memories = set()
        for query in keywords:
            relevant_memories.update(self._query_memory(query))
        relevant_memories = list(relevant_memories)
        relevant_memories = sorted(
            relevant_memories,
            key=lambda mem: datetime.datetime.strptime(
                re.search(r"\[(.*?)\]", mem).group(1), "%d %b %Y %H:%M:%S"
            ),
        )
        self._log.update({"Retrieved Memories": relevant_memories})
        hard_facts_prompt = interactive_document.InteractiveDocument(
            self._model
        )
        hard_facts_prompt.statement("\n".join(relevant_memories))
        hard_facts_question = (
            "Using only the above, determine the hard facts about the world "
            f"that {agent_name} is in."
        )
        hard_facts = hard_facts_prompt.open_question(
            hard_facts_question,
            max_tokens=1000,
            terminators=(),
            question_label="Exercise",
        ).rstrip("\n")
        self._log.update(
            {
                "Hard Facts CoT": hard_facts_prompt.view().text().splitlines(),
                "Hard Facts": hard_facts,
            }
        )
        prompt.statement("\n\n")
        prompt.statement("\n".join(relevant_memories))
        question = (
            "The search returned the above memories. Thoroughly analyse the "
            "above memories for answers to your "
            "questions. Be concise and ensure you are factually accurate."
        )
        answers = prompt.open_question(
            question,
            max_tokens=2000,
            terminators=(),
            question_label="Exercise",
        ).rstrip("\n")
        self._log.update(
            {
                "Retrieved Memories": relevant_memories,
                "Answers CoT": prompt.view().text().splitlines()[CoT_break:],
                "Hard facts": hard_facts,
                "Answers": answers,
            }
        )
        self._logging_channel(self._log)
        self._log = {"Key": self._pre_act_key}

        return hard_facts, answers, relevant_memories
