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


class Coordination(action_spec_ignored.ActionSpecIgnored):
    """
    Component for suggesting how those involved in a situation could
    coordinate to obtain the most prefereable outcome for the most number of
    people.
    """

    def __init__(
        self,
        model: language_model.LanguageModel,
        pre_act_key: str,
        clock,
        people_manager_name: entity_component.ComponentName,
        current_situation_name: entity_component.ComponentName,
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
        """
        super().__init__(pre_act_key)
        self._model = model
        self._people_manager_name = people_manager_name
        self._current_situation_name = current_situation_name
        self._memory_component_name = memory_component_name
        self._num_memories_to_retrieve = num_memories_to_retrieve
        self._clock = clock
        self._logging_channel = logging_channel
        self._log = {}
        self._previous_time = None

    def _make_pre_act_value(self):
        people_manager = self.get_entity().get_component(
            self._people_manager_name, type_=action_spec_ignored.ActionSpecIgnored
        )
        current_situation = self.get_entity().get_component(
            self._people_manager_name, type_=action_spec_ignored.ActionSpecIgnored
        )
