from collections.abc import Mapping
import types
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


class SuggestedAction(
    entity_component.ContextComponent, metaclass=abc.ABCMeta
):
    """A component that uses CoT to suggest a next action given the action
    spec.
    """

    def __init__(
        self,
        model: language_model.LanguageModel,
        high_level_plan: entity_component.ComponentName,
        current_situation: entity_component.ComponentName,
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
        self._high_level_plan = high_level_plan
        self._current_situation = current_situation
        self._observation = observation
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

        high_level_plan = (
            f"{agent_name}'s high-level plan:\n"
            f"{self.get_named_component_pre_act_value(self._high_level_plan)}"
            "\n\n"
        )
        current_situation_output = self.get_named_component_pre_act_value(
            self._current_situation
        )
        current_situation = (
            f"{agent_name}'s current situation:\n"
            f"{current_situation_output}"
            "\n\n"
        )

        observation = (
            f"{agent_name}'s recent observations:\n"
            f"{self.get_named_component_pre_act_value(self._observation)}"
            "\n\n"
        )

        prompt = interactive_document.InteractiveDocument(self._model)
        prompt.statement(f"{component_states}\n\n")
        prompt.statement(high_level_plan)
        prompt.statement(current_situation)
        prompt.statement(observation)

        if self._clock is not None:
            prompt.statement(f"Current time: {self._clock.now()}.\n")

        call_to_action = action_spec.call_to_action.format(
            name=agent_name,
            timedelta=helper_functions.timedelta_to_readable_str(
                self._clock.get_step_size()
            ),
        )
        prompt.statement(f"\nCall to action: {call_to_action}")

        if action_spec.output_type == entity_lib.OutputType.FREE:
            options_prompt = ""
        elif action_spec.output_type == entity_lib.OutputType.CHOICE:
            prompt.statement("\n".join(action_spec.options))
            options_prompt = (
                f" {agent_name} can only respond with one of the "
                "options provided by the call to action."
            )
        elif action_spec.output_type == entity_lib.OutputType.FLOAT:
            options_prompt = (
                f" {agent_name} can only respond with a single "
                "floating point number."
            )
        prompt.statement("\n")
        question = (
            f"Using the high-level plan and recent observations, suggest how "
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

        suggested_action = prompt.open_question(
            question,
            max_tokens=2000,
            terminators=(),
            question_label="Exercise",
        ).rstrip("\n")

        suggested_action_start = suggested_action.rfind("Suggestion:") + 11
        suggested_action = suggested_action[suggested_action_start:]

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
