import datetime
from collections.abc import Callable, Sequence, Mapping
import functools
import types
import abc
import threading

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

from concordia.factory.agent.custom_components.situations import CurrentSituation
from concordia.factory.agent.custom_components.suggested_action import SuggestedAction
from concordia.factory.agent.custom_components.hierarchical_plan import HierarchicalPlan
from concordia.factory.agent.custom_components.principle_shards import PrincipleShards
from concordia.factory.agent.custom_components.people_manager import PeopleManager
from concordia.factory.agent.custom_components.select_context_concat_act_component import SelectContextConcatActComponenet
from concordia.factory.agent.custom_components.questions import Questions

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
