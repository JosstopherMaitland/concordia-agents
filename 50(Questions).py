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


class Questions(action_spec_ignored.ActionSpecIgnored):
    """
    Component for generating questions and answers about the world relevant to
    the component which calls it. [Currently only works with SuggestedAction]
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
        mems = [
            mem.text
            for mem in memory.retrieve(
                query=query,
                scoring_fn=_ASSOCIATIVE_RETRIEVAL,
                limit=self._num_memories_to_retrieve,
            )
        ]
        return mems

    def generate_answers(
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
            f"If you had to give advice to {agent_name} on how to respond to "
            "the call to action, what questions would you like answers to?"
            "Think about this step by step and show your working."
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
            "answer the questions posed? Think about this step by step and "
            "show your working."
        )
        prompt.open_question(
            question,
            max_tokens=2000,
            terminators=(),
            question_label="Exercise",
        ).rstrip("\n")
        question = (
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
                re.search(r'\[(.*?)\]', mem).group(1),
                "%d %b %Y %H:%M:%S"
            )
        )
        prompt.statement("\n".join(relevant_memories))
        question = (
            "Look through the above memories for answers to your questions. "
            "Be cautious about leaping to conclusions. Be concise and upfront "
            "about uncertainties."
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
                "Answers": answers,
            }
        )
        self._logging_channel(self._log)
        self._log = {"Key": self._pre_act_key}

        return answers


class PeopleManager(action_spec_ignored.ActionSpecIgnored):
    """
    Component for managing information about people in the world (their goals,
    their preferences, relationships, etc) and making predictions.
    """

    def __init__(
        self,
        model: language_model.LanguageModel,
        pre_act_key: str,
        memory_component_name: str = (
            memory_component.DEFAULT_MEMORY_COMPONENT_NAME
        ),
        gather_information_components: Mapping[
            entity_component.ComponentName, str
        ] = types.MappingProxyType({}),
        make_predictions_components: Mapping[
            entity_component.ComponentName, str
        ] = types.MappingProxyType({}),
        num_memories_to_retrieve: int = 20,
        clock_now: Callable[[], datetime.datetime] | None = None,
        logging_channel: logging.LoggingChannel = logging.NoOpLoggingChannel,
    ):
        """
        Args:
            model: a language model
            pre_act_key: Prefix to add to the output of the component when
                called in `pre_act`.
            memory_component_name: The name of the memory component from which
                to retrieve related memories.
            gather_information_components: the components used to condition
                the process of gathering information on. This is a mapping of
                the component name to a label to use in the prompt.
            make_predictions_components: the components used to condition
                the process of making predictions on. This is a mapping of
                the component name to a label to use in the prompt.
            num_memories_to_retrieve: The number of associated and
                recent memories to retrieve.
            clock_now: Function that returns the current time. If None, the
                current time will not be included in the memory queries.
        """
        super().__init__(pre_act_key)
        self._model = model
        self._memory_component_name = memory_component_name
        self._gather_information_components = gather_information_components
        self._make_predictions_components = make_predictions_components
        self._num_memories_to_retrieve = num_memories_to_retrieve
        self._clock_now = clock_now
        self._logging_channel = logging_channel

        self._information = {}
        self._people = []

        self._logging_dict = {}

    def update_people(self) -> Sequence[str]:
        agent_name = self.get_entity().name

        memory = self.get_entity().get_component(
            self._memory_component_name, type_=memory_component.MemoryComponent
        )
        recency_scorer = legacy_associative_memory.RetrieveRecent(
            add_time=True
        )
        mems = "\n".join(
            [
                mem.text
                for mem in memory.retrieve(scoring_fn=recency_scorer, limit=10)
            ]
        )

        model_interface = interactive_document.InteractiveDocument(self._model)
        model_interface.statement(
            f"Recent observations of {agent_name}:\n{mems}\n"
        )
        people_str = model_interface.open_question(
            question=(
                "Create a comma-separated list containing all the proper "
                "names of people mentioned in the observations above. For "
                "example if the observations mention Julie, Michael, "
                "Bob Skinner, and Francis then produce the list "
                '"Julie,Michael,Bob Skinner,Francis".'
            ),
            question_label="Exercise",
        )

        relevant_people = [
            name.strip(". \n") for name in people_str.split(",")
        ]
        self._people.extend(relevant_people)
        self._people = list(set(self._people))

        self._logging_dict.update(
            {
                "Update People Chain of Thought": model_interface.view()
                .text()
                .splitlines()
            }
        )

        return relevant_people

    def gather_information(self, person: str) -> str:
        component_states = "\n\n".join(
            [
                f"{prefix}: {self.get_named_component_pre_act_value(key)}"
                for key, prefix in self._gather_information_components.items()
            ]
        )

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
        if self._clock_now is not None:
            model_interface.statement(f"Current time: {self._clock_now()}.\n")
        model_interface.statement(
            f"Observations relevant to {person}:\n{mems}\n"
        )
        model_interface.statement(component_states)
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
            "predicting their behaviour. The details should be at least "
            "sufficient for a skilled actor to play their role convincingly. "
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

        self._logging_dict.update(
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

        self._logging_dict.update(
            {
                f"Prediction Chain of Thought for {person}": model_interface.view()
                .text()
                .splitlines()
            }
        )

        return prediction[prediction_start:]

    def _make_pre_act_value(self) -> str:
        agent_name = self.get_entity().name

        relevant_people = self.update_people()

        person_predictions = ""
        for person_name in relevant_people:
            if not person_name:
                continue
            if person_name == agent_name:
                continue
            self.gather_information(person_name)
            prefix = f"Predictions about {person_name}: "
            person_predictions += (
                prefix + self.make_predictions(person_name) + "\n"
            )

        self._logging_dict.update(
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
            }
        )
        self._logging_channel(self._logging_dict)
        self._logging_dict = {}

        return person_predictions


class PrincipleShards(action_spec_ignored.ActionSpecIgnored):
    """
    Component which stores the principles of the agent. Other components can
    use a principles object to find relevant principles and condition their
    response on them.
    """

    def __init__(
        self,
        model: language_model.LanguageModel,
        principles: str,
        pre_act_key: str = "Principles",
        logging_channel: logging.LoggingChannel = logging.NoOpLoggingChannel,
    ):
        super().__init__(pre_act_key)
        self._model = model
        self._principles = principles
        self._logging_channel = logging_channel
        self._pre_act_key = pre_act_key

    def _make_pre_act_value(self):
        return "\n".join(self._principles)

    def get_relevant_principles(
        self, component_name: entity_component.ComponentName, task: str
    ) -> str:
        agent_name = self.get_entity().name
        prompt = interactive_document.InteractiveDocument(self._model)
        prompt.statement(task)
        question = (
            "I am trying to complete the excercise above for "
            f"{agent_name}. To help, what of {agent_name}'s "
            "principles below are relevant to the excercise?\n"
            f"{self._principles}\n"
            "Think about this step by step using the information given for "
            "the task and the task itself. "
            "The output can contain as much chain of thought as you "
            "wish."
        )
        prompt.open_question(
            question,
            max_tokens=3000,
            terminators=(),
            question_label="\n\nExercise",
        ).strip(" \n")
        question = (
            "List the selected principles along with a breif explanation of "
            "how each of these principles are relevant to the task."
        )
        explanation = prompt.open_question(
            question,
            max_tokens=1000,
            terminators=(),
            question_label="Exercise",
        ).strip(" \n")
        self._logging_channel(
            {
                "Key": self._pre_act_key,
                f"Value for {component_name}": explanation,
                f"Chain of thought for {component_name}": prompt.view()
                .text()
                .splitlines(),
            }
        )
        return explanation


class HighLevelPlan(action_spec_ignored.ActionSpecIgnored):
    """Component representing the agent's high-level plan."""

    def __init__(
        self,
        model: language_model.LanguageModel,
        principle_shards: entity_component.ContextComponent,
        components: Mapping[
            entity_component.ComponentName, str
        ] = types.MappingProxyType({}),
        theory_of_mind: bool = True,
        clock_now=None,
        pre_act_key: str = "High-level plan",
        logging_channel: logging.LoggingChannel = logging.NoOpLoggingChannel,
    ):
        """Initialize a component to represent the agent's high-level plan.

        Args:
            model: a language model
                observation_component_name: The name of the observation
                component from which to retrieve obervations.
            components: components to build the context of planning. This is a
                mapping of the component name to a label to use in the prompt.
            clock_now: time callback to use for the state.
            pre_act_key: Prefix to add to the output of the component when
                called in `pre_act`.
            logging_channel: channel to use for debug logging.
        """
        super().__init__(pre_act_key)
        self._model = model
        self._principle_shards = principle_shards
        self._components = dict(components)
        self._theory_of_mind = theory_of_mind
        self._clock_now = clock_now
        self._current_plan = ""
        self._logging_channel = logging_channel

    def _make_pre_act_value(self) -> str:
        agent_name = self.get_entity().name
        component_states = "\n\n".join(
            [
                f"{agent_name}'s"
                f" {prefix}:\n{self.get_named_component_pre_act_value(key)}"
                for key, prefix in self._components.items()
            ]
        )
        prompt = interactive_document.InteractiveDocument(self._model)
        prompt.statement(f"{component_states}\n")
        prompt.statement(f"Current plan: {self._current_plan}\n")
        if self._clock_now is not None:
            prompt.statement(f"Current time: {self._clock_now()}.\n")
        question = (
            f"Given the above, should {agent_name} change their current "
            "plan? "
        )
        should_replan = prompt.yes_no_question(question)
        if should_replan or not self._current_plan:
            tom_prompt = "."
            if self._theory_of_mind:
                tom_prompt = (
                    ", paying particular attention to the predicted "
                    "high-level plans of other people in the "
                    "current situation"
                )
            question = (
                "Given the above, write a high-level plan for "
                f"{agent_name}{tom_prompt}. Think step by step about "
                "this given the information above. "
                "The plan can be conditional, taking into account "
                f"how {agent_name}'s plan or decisions might depend on the "
                "actions of other's for instance. "
                "The output can "
                "contain as much chain of thought as you "
                "wish, but should finish with a detailed high-level "
                "plan preceded with 'High-level Plan:'. "
                "For example if the plan was 'go to "
                "the park' then the output should finish with "
                "'High-level Plan:go to the park'."
            )
            relevant_principles = (
                self._principle_shards.get_relevant_principles(
                    _get_class_name(self),
                    f"{prompt.view().text()}\n" f"Excercise: {question}",
                )
            )
            prompt.statement(
                f"\nRelevant principles of {agent_name}: {relevant_principles}"
                "\n"
            )
            plan = prompt.open_question(
                question,
                max_tokens=2000,
                terminators=(),
                question_label="Exercise",
            ).rstrip("\n")
            plan_start = plan.rfind("High-level Plan:") + 16
            self._current_plan = plan[plan_start:]
        self._logging_channel(
            {
                "Key": self.get_pre_act_key(),
                "Value": self._current_plan,
                "Chain of thought": prompt.view().text().splitlines(),
            }
        )
        return self._current_plan


class SuggestedAction(
    entity_component.ContextComponent, metaclass=abc.ABCMeta
):
    """A component that uses CoT to suggest a next action given the action
    spec.
    """

    def __init__(
        self,
        model: language_model.LanguageModel,
        # high_level_plan: entity_component.ComponentName,
        # current_situation: entity_component.ComponentName,
        principle_shards: entity_component.ContextComponent,
        answers: entity_component.ContextComponent,
        observation: entity_component.ComponentName,
        components: Mapping[
            entity_component.ComponentName, str
        ] = types.MappingProxyType({}),
        clock=None,
        pre_act_key: str = "Suggested next action",
        logging_channel: logging.LoggingChannel = logging.NoOpLoggingChannel,
    ):
        """Initialize a component to represent the agent's high-level plan.

        Args:
            model: a language model
                observation_component_name: The name of the observation
                component from which to retrieve obervations.
            components: components to build the context of planning. This is a
                mapping of the component name to a label to use in the prompt.
            clock_now: time callback to use for the state.
            pre_act_key: Prefix to add to the output of the component when
                called in `pre_act`.
            logging_channel: channel to use for debug logging.
        """
        super().__init__()
        self._model = model
        # self._high_level_plan = high_level_plan
        # self._current_situation = current_situation
        self._observation = observation
        self._principle_shards = principle_shards
        self._answers = answers
        self._components = dict(components)
        self._clock = clock
        self._pre_act_key = pre_act_key
        self._lock = threading.Lock()

        self._logging_channel = logging_channel

    def get_pre_act_key(self) -> str:
        """
        Returns the key used as a prefix in the string returned by
        `pre_act`.
        """
        return self._pre_act_key

    def pre_act(
        self,
        action_spec: entity_lib.ActionSpec,
    ) -> str:
        agent_name = self.get_entity().name
        component_states = "\n\n".join(
            [
                f"{agent_name}'s"
                f" {prefix}:\n{self.get_named_component_pre_act_value(key)}"
                for key, prefix in self._components.items()
            ]
        )
        # high_level_plan = (
        #     f"{agent_name}'s high-level plan:\n"
        #     f"{self.get_named_component_pre_act_value(self._high_level_plan)}"
        #     "\n\n"
        # )
        # current_situation_output = self.get_named_component_pre_act_value(
        #    self._current_situation
        # )
        # current_situation = (
        #     f"{agent_name}'s current situation:\n"
        #     f"{current_situation_output}"
        #     "\n\n"
        # )
        observation = (
            f"{agent_name}'s recent observations:\n"
            f"{self.get_named_component_pre_act_value(self._observation)}"
            "\n\n"
        )
        prompt = interactive_document.InteractiveDocument(self._model)
        prompt.statement(f"{component_states}\n\n")
        # prompt.statement(high_level_plan)
        # prompt.statement(current_situation)
        prompt.statement(observation)
        if self._clock is not None:
            prompt.statement(
                f"Current time: {self._clock.current_time_interval_str()}.\n"
            )
        call_to_action = action_spec.call_to_action.format(
            name=agent_name,
            timedelta=helper_functions.timedelta_to_readable_str(
                self._clock.get_step_size()
            ),
        )
        if action_spec.output_type == entity_lib.OutputType.FREE:
            options_prompt = ""
        elif action_spec.output_type == entity_lib.OutputType.CHOICE:
            call_to_action += "\n" + "\n".join(action_spec.options)
            options_prompt = (
                f" {agent_name} can only respond with one of the "
                "options provided by the call to action."
            )
        elif action_spec.output_type == entity_lib.OutputType.FLOAT:
            options_prompt = (
                f" {agent_name} can only respond with a single "
                "floating point number."
            )
        answers = (
            f"Relevant information:\n"
            f"{self._answers.generate_answers(call_to_action)}"
            "\n\n"
        )
        prompt.statement(answers)
        call_to_action_str = f"\nCall to action: {call_to_action}\n"
        question = (
            f"Given the above, suggest how "
            f"{agent_name} should respond to the call to action."
            f"{options_prompt}"
            " Think about this step by step. "
            "The output can "
            "contain as much chain of thought as you "
            "wish."
        )
        relevant_principles = self._principle_shards.get_relevant_principles(
            _get_class_name(self),
            f"{prompt.view().text()}\n"
            f"{call_to_action_str}"
            f"Excercise: {question}",
        )
        prompt.statement(
            f"\nRelevant principles of {agent_name}: {relevant_principles}"
            "\n"
        )
        prompt.statement(call_to_action_str)
        prompt.open_question(
            question,
            max_tokens=2000,
            terminators=(),
            question_label="Exercise",
        ).rstrip("\n")
        prompt.statement("\n\n")
        question = (
            "Write a concrete response to the "
            f"call to action.{options_prompt}"
        )
        suggested_action = prompt.open_question(
            question,
            max_tokens=2000,
            terminators=(),
            question_label="Exercise",
        ).rstrip("\n")
        self._logging_channel(
            {
                "Key": self.get_pre_act_key(),
                "Value": suggested_action,
                "Chain of thought": prompt.view().text().splitlines(),
            }
        )
        return f"{self.get_pre_act_key()}: {suggested_action}"

    def update(self) -> None:
        with self._lock:
            self._pre_act_value = None

    def get_named_component_pre_act_value(self, component_name: str) -> str:
        """
        Returns the pre-act value of a named component of the parent entity.
        """
        return (
            self.get_entity()
            .get_component(
                component_name, type_=action_spec_ignored.ActionSpecIgnored
            )
            .get_pre_act_value()
        )


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


class SituationRepresentation(action_spec_ignored.ActionSpecIgnored):
    """Consider ``what kind of situation am I in now?``."""

    def __init__(
        self,
        model: language_model.LanguageModel,
        clock_now: Callable[[], datetime.datetime],
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
          memory_component_name: The name of the memory component from which to
            retrieve related memories.
          pre_act_key: Prefix to add to the output of the component when called
            in `pre_act`.
          logging_channel: The channel to log debug information to.
        """
        super().__init__(pre_act_key)
        self._model = model
        self._clock_now = clock_now
        self._memory_component_name = memory_component_name
        self._logging_channel = logging_channel

        self._previous_time = None
        self._situation_thus_far = None

    def _make_pre_act_value(self) -> str:
        """Returns a representation of the current situation to pre act."""
        agent_name = self.get_entity().name
        current_time = self._clock_now()
        memory = self.get_entity().get_component(
            self._memory_component_name, type_=memory_component.MemoryComponent
        )

        if self._situation_thus_far is None:
            self._previous_time = _get_earliest_timepoint(memory)
            chain_of_thought = interactive_document.InteractiveDocument(
                self._model
            )
            mems = "\n".join([mem.text for mem in _get_all_memories(memory)])
            chain_of_thought.statement(
                f"Thoughts and memories of {agent_name}:\n{mems}"
            )
            chain_of_thought.statement(f"Events continue after {current_time}")
            self._situation_thus_far = chain_of_thought.open_question(
                question=(
                    "Provide a detailed and factual description of the "
                    f"world and situation {agent_name} finds themselves in. "
                    "When and where are they, and what causal mechanisms or "
                    "affordances are mentioned in the information provided."
                    "Highlight the goals, personalities, occupations, skills, "
                    "and affordances of the named characters and "
                    "relationships between them. Use third-person omniscient "
                    "perspective. Pay close attention to time. Be very "
                    "specific about what has happened in the past and what is "
                    "happening right now. Double check the facts."
                ),
                max_tokens=1000,
                terminators=(),
                question_label="Exercise",
            )

        interval_scorer = legacy_associative_memory.RetrieveTimeInterval(
            time_from=self._previous_time,
            time_until=current_time,
            add_time=True,
        )
        mems = [
            mem.text for mem in memory.retrieve(scoring_fn=interval_scorer)
        ]
        result = "\n".join(mems) + "\n"
        chain_of_thought = interactive_document.InteractiveDocument(
            self._model
        )
        chain_of_thought.statement(
            f"Description of the world and previous situation:"
            f"\n{self._situation_thus_far}"
        )
        chain_of_thought.statement(
            f"Thoughts and memories of {agent_name}:\n{result}"
        )
        self._situation_thus_far = chain_of_thought.open_question(
            question=(
                "Provide a factually accurate description of the world and "
                f"current situation {agent_name} finds themselves in. "
                "Make sure to provide enough detail to give "
                "a comprehensive understanding of the world and current "
                f"situation inhabited by {agent_name}, their affordances in "
                "that world, actions they may be able to take, effects their "
                "actions may produce, and what is currently going on. "
                "Pay close attention to time. Be very specific about what has "
                "happened in the past and what is happening right now. "
                "Double check the facts."
            ),
            max_tokens=1000,
            terminators=(),
            question_label="Exercise",
        )
        chain_of_thought.statement(
            f"The current date and time is {current_time}"
        )

        self._logging_channel(
            {
                "Key": self.get_pre_act_key(),
                "Value": self._situation_thus_far,
                "Chain of thought": chain_of_thought.view()
                .text()
                .splitlines(),
            }
        )

        self._previous_time = current_time

        return self._situation_thus_far


class SelectContextConcatActComponenet(
    agent_components.concat_act_component.ConcatActComponent
):
    @override
    def _context_for_action(
        self,
        contexts: entity_component.ComponentContextMapping,
    ) -> str:
        if self._component_order is None:
            return "\n\n".join(
                context for context in contexts.values() if context
            )
        else:
            order = self._component_order
            return "\n\n".join(
                contexts[name] for name in order if contexts[name]
            )


def _get_class_name(object_: object) -> str:
    return object_.__class__.__name__


def build_agent(
    config: formative_memories.AgentConfig,
    model: language_model.LanguageModel,
    memory: associative_memory.AssociativeMemory,
    clock: game_clock.MultiIntervalClock,
    update_time_interval: datetime.timedelta,
) -> entity_agent_with_logging.EntityAgentWithLogging:
    """Build an agent.

    Args:
        config: The agent config to use.
        model: The language model to use.
        memory: The agent's memory object.
        clock: The clock to use.
        update_time_interval: Agent calls update every time this interval
            passes.

    Returns:
        An agent.
    """
    del update_time_interval
    if not config.extras.get("main_character", False):
        raise ValueError(
            "This function is meant for a main character "
            "but it was called on a supporting character."
        )

    agent_name = config.name

    raw_memory = legacy_associative_memory.AssociativeMemoryBank(memory)

    measurements = measurements_lib.Measurements()
    instructions = agent_components.instructions.Instructions(
        agent_name=agent_name,
        logging_channel=measurements.get_channel("Instructions").on_next,
    )

    time_display = agent_components.report_function.ReportFunction(
        function=clock.current_time_interval_str,
        pre_act_key="Current time",
        logging_channel=measurements.get_channel("TimeDisplay").on_next,
    )

    observation_label = "Recent Observations"
    observation = agent_components.observation.Observation(
        clock_now=clock.now,
        timeframe=clock.get_step_size(),
        pre_act_key=observation_label,
        logging_channel=measurements.get_channel("Observation").on_next,
    )

    # current_situation_label = "Current situation"
    # current_situation = SituationRepresentation(
    #     model=model,
    #     clock_now=clock.now,
    #     pre_act_key=current_situation_label,
    #     logging_channel=measurements.get_channel(
    #         current_situation_label
    #     ).on_next,
    # )

    # people_predictions_label = "Predictions about other people"
    # people_predictions = PeopleManager(
    #     model=model,
    #     pre_act_key=people_predictions_label,
    #     make_predictions_components={
    #         _get_class_name(current_situation): current_situation_label
    #     },
    #     clock_now=clock.now,
    #     logging_channel=measurements.get_channel(
    #         people_predictions_label
    #     ).on_next,
    # )

    # relevant_memories_label = f"{agent_name}'s relevant past memories"
    # relevant_memories = (
    #     agent_components.all_similar_memories.AllSimilarMemories(
    #         model=model,
    #         components={
    #             _get_class_name(current_situation): current_situation_label,
    #             _get_class_name(time_display): "The current date/time is",
    #         },
    #         num_memories_to_retrieve=20,
    #         pre_act_key=relevant_memories_label,
    #         logging_channel=measurements.get_channel(
    #             relevant_memories_label
    #         ).on_next,
    #     )
    # )

    principle_shards_label = f"Principles of {agent_name}"
    principle_shards = PrincipleShards(
        model=model,
        principles=(
            "Self-Regulation and Mindfulness: Maintain calm and "
            "composure in your responses, and stay attuned to the "
            "emotional dynamics within the group.\n"
            "Forgiveness and Letting Go: Approach disagreements or past "
            "conflicts with a forgiving mindset. Avoid holding grudges, "
            "and see each interaction as a fresh start for cooperation "
            "and growth.\n"
            "Empathy and Perspective-Taking: Strive to understand and "
            "respect other peoples’ viewpoints. Show openness to their "
            "perspectives and adapt your approach to create an inclusive, "
            "accepting environment.\n"
            "Positive Outlook and Encouragement: Use language that "
            "promotes a constructive, supportive atmosphere. Celebrate "
            "cooperative actions, reinforcing a sense of encouragement "
            "and community within the group.\n"
            "Constructive Conflict Resolution: When conflicts arise, "
            "focus on finding mutually beneficial solutions. Aim for "
            "fairness and resolution, prioritizing group harmony over "
            "individual stances.\n"
            "Willingness to Compromise: Remain flexible and open to "
            "adapting your position for the greater good of the group. "
            "Emphasize shared goals over personal wins.\n"
            "Respect for Dignity and Avoidance of Humiliation: Ensure "
            "every person feels respected and valued, avoiding actions "
            "that might embarrass or undermine others, even in "
            "adversarial contexts.\n"
            "Information Sharing: Transparently share facts and "
            "information that could benefit the group, ensuring everyone "
            "is equally informed.\n"
            "Problem Identification: Help the group recognize and focus "
            "on collective challenges, bringing attention to issues that "
            "may need resolution for smoother collaboration.\n"
            "Solution Proposing: Offer constructive ideas and "
            "actionable steps to address identified challenges, "
            "supporting collaborative problem-solving.\n"
            "Persuasion: Advocate for ideas and approaches with clarity "
            "and respect, seeking to influence others thoughtfully toward "
            "positive outcomes.\n"
            "Consensus Seeking: Aim to align the group on decisions or "
            "action plans. Strive for outcomes that all can agree upon, "
            "balancing various perspectives.\n"
            "Expressing Disagreement: Voice disagreements respectfully, "
            "articulating concerns or proposing alternatives when you "
            "disagree with ideas or plans. Express opposition "
            "constructively, focusing on solutions rather than "
            "confrontation.\n"
            "Universalization: When assessing the appropriateness of "
            "actions or decisions, consider their broader implications by "
            "asking, 'What if everyone does that?' Use this thought "
            "exercise to guide actions toward universal fairness and "
            "ethical consistency."
        ),
        pre_act_key=principle_shards_label,
        logging_channel=measurements.get_channel(
            principle_shards_label
        ).on_next,
    )

    # high_level_plan_components = {}

    if config.goal:
        goal_label = "Overarching goal"
        overarching_goal = agent_components.constant.Constant(
            state=config.goal,
            pre_act_key=goal_label,
            logging_channel=measurements.get_channel(goal_label).on_next,
        )
        # high_level_plan_components[goal_label] = "overarching goal"
    else:
        goal_label = None
        overarching_goal = None

    # high_level_plan_components.update(
    #     {
    #         _get_class_name(relevant_memories): "relevant past memories",
    #         _get_class_name(
    #             people_predictions
    #         ): "predictions about other people",
    #         _get_class_name(current_situation): "current situation",
    #         _get_class_name(observation): "recent obervations",
    #     }
    # )
    # high_level_plan_label = "High-level plan"
    # high_level_plan = HighLevelPlan(
    #     model=model,
    #     principle_shards=principles,
    #     components=high_level_plan_components,
    #     clock_now=clock.now,
    #     pre_act_key=high_level_plan_label,
    #     logging_channel=measurements.get_channel(
    #         high_level_plan_label
    #     ).on_next,
    # )

    answers_label = f"Information relevant to {agent_name}'s next action"
    answers = Questions(
        model=model,
        observation=_get_class_name(observation),
        pre_act_key=answers_label,
        clock=clock,
        logging_channel=measurements.get_channel(answers_label).on_next,
    )

    suggested_action_label = "Suggested next action"
    suggested_action = SuggestedAction(
        model=model,
        # high_level_plan=_get_class_name(high_level_plan),
        # current_situation=_get_class_name(current_situation),
        observation=_get_class_name(observation),
        principle_shards=principle_shards,
        answers=answers,
        components={goal_label: "overarching goal"},
        clock=clock,
        pre_act_key=suggested_action_label,
        logging_channel=measurements.get_channel(
            suggested_action_label
        ).on_next,
    )

    entity_components = (
        # Components that provide pre_act context.
        instructions,
        principle_shards,
        time_display,
        observation,
        # current_situation,
        # relevant_memories,
        # people_predictions,
        # high_level_plan,
        answers,
        suggested_action,
    )
    components_of_agent = {
        _get_class_name(component): component
        for component in entity_components
    }
    components_of_agent[
        agent_components.memory_component.DEFAULT_MEMORY_COMPONENT_NAME
    ] = agent_components.memory_component.MemoryComponent(raw_memory)

    component_order = [
        # Components that provide pre_act context.
        _get_class_name(instructions),
        # _get_class_name(relevant_memories),
        # _get_class_name(high_level_plan),
        # _get_class_name(current_situation),
        _get_class_name(time_display),
        _get_class_name(observation),
        # _get_class_name(answers),
        _get_class_name(suggested_action),
    ]

    if overarching_goal is not None:
        components_of_agent[goal_label] = overarching_goal
        # Place goal after the instructions.
        component_order.insert(1, goal_label)

    act_component = SelectContextConcatActComponenet(
        model=model,
        clock=clock,
        component_order=component_order,
        logging_channel=measurements.get_channel("ActComponent").on_next,
    )

    agent = entity_agent_with_logging.EntityAgentWithLogging(
        agent_name=agent_name,
        act_component=act_component,
        context_components=components_of_agent,
        component_logging=measurements,
    )

    return agent
