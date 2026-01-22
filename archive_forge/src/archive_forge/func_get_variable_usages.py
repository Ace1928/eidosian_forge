from ..language.ast import (FragmentDefinition, FragmentSpread,
from ..language.visitor import ParallelVisitor, TypeInfoVisitor, Visitor, visit
from ..type import GraphQLSchema
from ..utils.type_info import TypeInfo
from .rules import specified_rules
def get_variable_usages(self, node):
    usages = self._variable_usages.get(node)
    if usages is None:
        usages = []
        sub_visitor = UsageVisitor(usages, self._type_info)
        visit(node, TypeInfoVisitor(self._type_info, sub_visitor))
        self._variable_usages[node] = usages
    return usages