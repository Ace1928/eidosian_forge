from ..language.ast import (FragmentDefinition, FragmentSpread,
from ..language.visitor import ParallelVisitor, TypeInfoVisitor, Visitor, visit
from ..type import GraphQLSchema
from ..utils.type_info import TypeInfo
from .rules import specified_rules
def get_recursively_referenced_fragments(self, operation):
    assert isinstance(operation, OperationDefinition)
    fragments = self._recursively_referenced_fragments.get(operation)
    if not fragments:
        fragments = []
        collected_names = set()
        nodes_to_visit = [operation.selection_set]
        while nodes_to_visit:
            node = nodes_to_visit.pop()
            spreads = self.get_fragment_spreads(node)
            for spread in spreads:
                frag_name = spread.name.value
                if frag_name not in collected_names:
                    collected_names.add(frag_name)
                    fragment = self.get_fragment(frag_name)
                    if fragment:
                        fragments.append(fragment)
                        nodes_to_visit.append(fragment.selection_set)
        self._recursively_referenced_fragments[operation] = fragments
    return fragments