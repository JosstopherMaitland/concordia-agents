
from concordia.components import agent as agent_components
from concordia.typing import entity_component
from typing_extensions import override


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
