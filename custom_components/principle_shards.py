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
            question_label="\n\nExercise",
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
