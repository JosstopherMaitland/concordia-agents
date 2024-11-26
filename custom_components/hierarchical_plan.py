import abc
import threading

from concordia.document import interactive_document
from concordia.language_model import language_model
from concordia.memory_bank import legacy_associative_memory
from concordia.typing import entity_component
from concordia.typing import entity as entity_lib
from concordia.typing import logging
from concordia.components.agent import action_spec_ignored
from concordia.utils import helper_functions

_ASSOCIATIVE_RETRIEVAL = legacy_associative_memory.RetrieveAssociative()


def _get_class_name(object_: object) -> str:
    return object_.__class__.__name__


class HierarchicalPlan(
    entity_component.ContextComponent, metaclass=abc.ABCMeta
):
    """
    A component that uses a hierarchical planning CoT to suggest a next
    action given the action spec.
    """

    def __init__(
        self,
        model: language_model.LanguageModel,
        principle_shards: entity_component.ContextComponent,
        information: entity_component.ContextComponent,
        goal: entity_component.ComponentName,
        observation: entity_component.ComponentName,
        clock=None,
        pre_act_key: str = "Suggested next action",
        logging_channel: logging.LoggingChannel = logging.NoOpLoggingChannel,
    ):
        """
        Args:
            model: a language model
                observation_component_name: The name of the observation
                component from which to retrieve obervations.
            principle_shards: The component which decides which of the agent's
                principles are relevant. Currently required.
            clock: time callback to use for the state.
            pre_act_key: Prefix to add to the output of the component when
                called in `pre_act`.
            logging_channel: channel to use for debug logging.
        """
        super().__init__()
        self._model = model
        self._observation = observation
        self._principle_shards = principle_shards
        self._information = information
        self._goal = goal
        self._clock = clock
        self._pre_act_key = pre_act_key
        self._lock = threading.Lock()
        self._logging_channel = logging_channel
        self._longterm_plan = ""
        self._log = {"Key": pre_act_key}

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
        context_list = []
        observation_str = (
            f"{agent_name}'s recent observations:\n"
            f"{self.get_named_component_pre_act_value(self._observation)}"
        )
        call_to_action_str = action_spec.call_to_action.format(
            name=agent_name,
            timedelta=helper_functions.timedelta_to_readable_str(
                self._clock.get_step_size()
            ),
        )
        if action_spec.output_type == entity_lib.OutputType.FREE:
            options_prompt = ""
        elif action_spec.output_type == entity_lib.OutputType.CHOICE:
            call_to_action_str += "\n" + "\n".join(action_spec.options)
            options_prompt = (
                f" {agent_name} can only respond with one of the "
                "options provided by the call to action."
            )
        elif action_spec.output_type == entity_lib.OutputType.FLOAT:
            options_prompt = (
                f" {agent_name} can only respond with a single "
                "floating point number."
            )
        hard_facts, answers, _ = self._information.generate_information(
            call_to_action_str
        )
        information_str = (
            f"Information about the world:\n" f"{hard_facts}\n{answers}"
        )
        context_list.extend([information_str, observation_str])
        if self._goal:
            goal_str = (
                f"Overarching goal of {agent_name}:\n"
                f"{self.get_named_component_pre_act_value(self._goal)}"
            )
            context_list.insert(0, goal_str)
        if self._clock is not None:
            current_time_str = (
                f"Current time: {self._clock.current_time_interval_str()}."
            )
            context_list.append(current_time_str)
        # LONGTERM PLAN:
        longterm_prompt = interactive_document.InteractiveDocument(self._model)
        longterm_context_list = []
        longterm_context_list.extend(context_list)
        current_plan_str = (
            f"{agent_name}'s current longterm plan:\n" f"{self._longterm_plan}"
        )
        longterm_context_list.append(current_plan_str)
        longterm_context_str = "\n\n".join(longterm_context_list) + "\n\n"
        longterm_prompt.statement(longterm_context_str)
        question = (
            f"Given all the above, should {agent_name} change their current "
            "longterm plan?"
        )
        should_replan = longterm_prompt.yes_no_question(question)
        if should_replan or not self._longterm_plan:
            question = (
                "Given the above, write a longterm plan for "
                f"{agent_name}. It should be "
                f"farsighted and give {agent_name} the best possible chance "
                f"of fulfilling their goals. "
                "Think step by step about "
                "this given the information above. "
                "The plan can be conditional, taking into account "
                f"how {agent_name}'s plan or decisions might depend on the "
                "actions of other's for instance. "
                "The output can "
                "contain as much chain of thought as you "
                "wish, but should finish with a detailed longterm "
                "plan preceded with 'Longterm Plan:'. "
                "For example if the plan was 'go to "
                "the park' then the output should finish with "
                "'Longterm Plan:go to the park'."
            )
            relevant_principles = (
                self._principle_shards.get_relevant_principles(
                    _get_class_name(self),
                    f"{longterm_context_str}" f"Excercise: {question}",
                )
            )
            relevant_principles_str = (
                f"Relevant principles of {agent_name}:\n{relevant_principles}"
            )
            longterm_prompt.statement(f"\n\n{relevant_principles_str}\n\n")
            plan = longterm_prompt.open_question(
                question,
                max_tokens=2000,
                terminators=(),
                question_label="Exercise",
            ).rstrip("\n")
            plan_start = plan.rfind("Longterm Plan:") + 14
            self._longterm_plan = plan[plan_start:]
        self._log.update(
            {
                "Longterm Plan": self._longterm_plan,
                "Longterm Plan CoT": longterm_prompt.view()
                .text()
                .splitlines(),
            }
        )
        # SUGGESTED NEXT ACTION
        action_prompt = interactive_document.InteractiveDocument(self._model)
        action_prompt.statement("\n\n".join(context_list))
        current_plan_str = (
            f"{agent_name}'s current longterm plan:\n" f"{self._longterm_plan}"
        )
        action_prompt.statement(f"\n\n{current_plan_str}\n\n")
        call_to_action_str = f"Call to action:\n{call_to_action_str}"
        action_prompt.statement(f"\n\n{call_to_action_str}\n\n")
        question = (
            f"Given the above, suggest how "
            f"{agent_name} should respond to the call to action."
            f"{options_prompt}"
            " Think about this step by step."
            "The output can "
            "contain as much chain of thought as you "
            "wish, but should finish with a concrete response to the "
            "call to action preceded with 'Suggestion:'. "
            "For example if the suggested response was 'go to "
            "the park' then the output should finish with "
            "'Suggestion:go to the park'."
        )
        suggested_action = action_prompt.open_question(
            question,
            max_tokens=2000,
            terminators=(),
            question_label="Exercise",
        ).rstrip("\n")
        suggested_action_start = suggested_action.rfind("Suggestion:") + 11
        suggested_action = suggested_action[suggested_action_start:]
        self._log.update(
            {
                "Suggested Action": suggested_action,
                "Suggested Action CoT": action_prompt.view()
                .text()
                .splitlines(),
            }
        )
        self._logging_channel(self._log)
        self._log = {"Key": self._pre_act_key}
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
