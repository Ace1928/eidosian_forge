from ..language.ast import (FragmentDefinition, FragmentSpread,
from ..language.visitor import ParallelVisitor, TypeInfoVisitor, Visitor, visit
from ..type import GraphQLSchema
from ..utils.type_info import TypeInfo
from .rules import specified_rules
def visit_using_rules(schema, type_info, ast, rules):
    context = ValidationContext(schema, ast, type_info)
    visitors = [rule(context) for rule in rules]
    visit(ast, TypeInfoVisitor(type_info, ParallelVisitor(visitors)))
    return context.get_errors()