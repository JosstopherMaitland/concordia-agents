import datetime
from collections.abc import Sequence

from concordia.components.agent import memory_component
from concordia.document import interactive_document
from concordia.components import agent as agent_components
from concordia.language_model import language_model
from concordia.memory_bank import legacy_associative_memory
from concordia.typing import logging
from concordia.components.agent import action_spec_ignored
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


class PeopleManager(action_spec_ignored.ActionSpecIgnored):
    """
    Component for managing information about people in the world (their goals,
    their preferences, relationships, etc) and making predictions.
    """

    def __init__(
        self,
        model: language_model.LanguageModel,
        pre_act_key: str,
        clock,
        memory_component_name: str = (
            memory_component.DEFAULT_MEMORY_COMPONENT_NAME
        ),
        num_memories_to_retrieve: int = 20,
        logging_channel: logging.LoggingChannel = logging.NoOpLoggingChannel,
    ):
        """
        Args:
            model: a language model
            pre_act_key: Prefix to add to the output of the component when
                called in `pre_act`.
            memory_component_name: The name of the memory component from which
                to retrieve related memories.
            make_predictions_components: the components used to condition
                the process of making predictions on. This is a mapping of
                the component name to a label to use in the prompt.
            num_memories_to_retrieve: The number of associated memories to
                retrieve.
            clock: game clock object for retrieving the current time.
        """
        super().__init__(pre_act_key)
        self._model = model
        self._memory_component_name = memory_component_name
        self._num_memories_to_retrieve = num_memories_to_retrieve
        self._clock = clock
        self._logging_channel = logging_channel
        self._information = {}
        self._preferences = {}
        self._people = []
        self._log = {}
        self._previous_time = None

    def update_people(self) -> Sequence[str]:
        agent_name = self.get_entity().name
        current_time = self._clock.now()
        memory = self.get_entity().get_component(
            self._memory_component_name, type_=memory_component.MemoryComponent
        )
        if self._previous_time is None:
            self._previous_time = _get_earliest_timepoint(memory)
            mems = "\n".join([mem.text for mem in _get_all_memories(memory)])
        else:
            interval_scorer = legacy_associative_memory.RetrieveTimeInterval(
                time_from=self._previous_time,
                time_until=current_time,
                add_time=True,
            )
            mems = [
                mem.text for mem in memory.retrieve(scoring_fn=interval_scorer)
            ]
        model_interface = interactive_document.InteractiveDocument(self._model)
        model_interface.statement(
            f"Recent observations of {agent_name}:\n{mems}\n"
        )
        question = (
            "Create a comma-separated list containing all the proper "
            "names of people mentioned in the observations above. For "
            "example if the observations mention Julie, Michael, "
            "Bob Skinner, and Francis then produce the list "
            "'Julie,Michael,Bob Skinner,Francis'."
        )
        people_str = model_interface.open_question(
            question=question,
            question_label="Exercise",
        )
        recent_people = [name.strip(". \n") for name in people_str.split(",")]
        self._people.extend(recent_people)
        self._people = list(set(self._people))
        self._log.update(
            {
                "Update People Chain of Thought": model_interface.view()
                .text()
                .splitlines()
            }
        )
        self._previous_time = current_time
        return recent_people

    def gather_information(self, person: str) -> str:
        memory = self.get_entity().get_component(
            self._memory_component_name, type_=memory_component.MemoryComponent
        )
        mems = "\n".join(
            [
                mem.text
                for mem in memory.retrieve(
                    query=person,
                    scoring_fn=_ASSOCIATIVE_RETRIEVAL,
                    limit=self._num_memories_to_retrieve,
                )
            ]
        )
        model_interface = interactive_document.InteractiveDocument(self._model)
        model_interface.statement(
            "Current date and time: "
            f"{self._clock.current_time_interval_str()}.\n"
        )
        model_interface.statement(
            f"Observations relevant to {person}:\n{mems}\n"
        )
        if person in self._information.keys():
            model_interface.statement(
                f"Previous information about {person}:"
                f"\n{self._information[person]}\n"
            )
            prev_info_prompt = (
                f"If parts of the previous information about {person}"
                "are useful and still true, then repeat them or "
                "iterate and refine them to better match "
                "the relevant observations so far."
            )
        else:
            prev_info_prompt = ""
        question = (
            f"Given the above information, identify key details about "
            f"{person} "
            f"that would be helpful in taking on their perspective and "
            "predicting their behaviour. "
            f"{prev_info_prompt} "
            "Details to extract or infer could include: \nGoals and "
            "Intentions. "
            "What does this person seem to want or aim to achieve? Are there "
            "hints about their short-term and long-term goals?\n"
            "Preferences and Priorities. What are this person’s likely "
            "preferences, values, or areas of focus? For example, do they "
            "prioritize personal gain, cooperation, or certain topics?\n"
            "Behavioral Patterns and Communication Style. How does this "
            "person "
            "typically act or communicate in interactions? Do they exhibit "
            "any "
            "habits, social cues, or tendencies that could be useful in "
            "predicting future behavior?\n"
            "Social Connections and Alliances. Does this person have any "
            "known affiliations, allies, or loyalties? How might these "
            "relationships influence their actions?\n"
            "Negotiation or Persuasion Leverage: Are there any "
            f"vulnerabilities, motivations, or incentives that "
            "could be used to influence this person’s decisions?\n"
        )
        person_information = model_interface.open_question(
            question,
            max_tokens=2000,
            question_label="Exercise",
            terminators=(),
        ).rstrip("\n")
        self._information[person] = person_information
        question = "Infer and state the goals and preferences of {person}."
        person_preferences = model_interface.open_question(
            question,
            max_tokens=2000,
            question_label="Exercise",
            terminators=(),
        ).rstrip("\n")
        self._preferences[person] = person_preferences
        self._log.update(
            {
                f"Gather Info Chain of Thought for {person}": (
                    model_interface.view().text().splitlines()
                )
            }
        )
        return person_information

    def make_predictions(self, person: str) -> str:
        memory = self.get_entity().get_component(
            self._memory_component_name, type_=memory_component.MemoryComponent
        )
        mems = "\n".join(
            [
                mem.text
                for mem in memory.retrieve(
                    query=person,
                    scoring_fn=_ASSOCIATIVE_RETRIEVAL,
                    limit=self._num_memories_to_retrieve,
                )
            ]
        )
        component_states = "\n\n".join(
            [
                f"{prefix}:\n{self.get_named_component_pre_act_value(key)}"
                for key, prefix in self._make_predictions_components.items()
            ]
        )
        model_interface = interactive_document.InteractiveDocument(self._model)
        model_interface.statement(
            f"Relevant observations of {person}:\n{mems}"
        )
        model_interface.statement(component_states)
        model_interface.statement(
            f"Information about {person}:\n{self._information[person]}"
        )
        if self._clock_now is not None:
            model_interface.statement(f"Current time: {self._clock_now()}.\n")
        question = (
            "Given the above, generate a prediction about "
            f"{person}'s high-level plan and how they are going to "
            "behave in the current situation.\n"
            "Think step by step about this given the information above."
            "The prediction can be conditional, taking into account "
            f"how {person}'s behaviour or decisions might depend on the "
            "actions of other's for instance."
            " The output can contain as much chain of thought as you "
            "wish, but should finish with a detailed prediction "
            "preceded with 'Prediction:'. "
            "For example if the prediction was 'they will go to "
            "the park' then the output should finish with 'Prediction:"
            "they will go to the park'."
        )
        prediction = model_interface.open_question(
            question,
            max_tokens=2000,
            question_label="Exercise",
            terminators=(),
        ).rstrip("\n")
        prediction_start = prediction.rfind("Prediction:") + 11
        self._log.update(
            {
                f"Prediction CoT for {person}": model_interface.view()
                .text()
                .splitlines()
            }
        )
        return prediction[prediction_start:]

    def _make_pre_act_value(self) -> str:
        agent_name = self.get_entity().name
        recent_people = self.update_people()
        person_predictions = ""
        for person_name in recent_people:
            if not person_name:
                continue
            if person_name == agent_name:
                continue
            self.gather_information(person_name)
            prefix = f"Predictions about {person_name}: "
            person_predictions += (
                prefix + self.make_predictions(person_name) + "\n"
            )
        self._log.update(
            {
                "Key": self.get_pre_act_key(),
                "Value": person_predictions,
                "All Names": ",".join(self._people),
                "All Information": "\n".join(
                    [
                        f"{person}: {info}"
                        for person, info in self._information.items()
                    ]
                ),
                "All Goals and Preferences": "\n".join(
                    [
                        f"{person}: {pref}"
                        for person, pref in self._preferences.items()
                    ]
                ),
            }
        )
        self._logging_channel(self._log)
        self._log = {}
        return person_predictions
