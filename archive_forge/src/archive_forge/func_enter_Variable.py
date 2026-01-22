from ..language.ast import (FragmentDefinition, FragmentSpread,
from ..language.visitor import ParallelVisitor, TypeInfoVisitor, Visitor, visit
from ..type import GraphQLSchema
from ..utils.type_info import TypeInfo
from .rules import specified_rules
def enter_Variable(self, node, key, parent, path, ancestors):
    usage = VariableUsage(node, type=self.type_info.get_input_type())
    self.usages.append(usage)