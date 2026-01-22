from ..language.ast import BooleanValue, FloatValue, IntValue, StringValue
from .definition import GraphQLScalarType
def parse_boolean_literal(ast):
    if isinstance(ast, BooleanValue):
        return ast.value
    return None