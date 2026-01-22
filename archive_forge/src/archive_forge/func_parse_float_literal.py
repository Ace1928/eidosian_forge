from ..language.ast import BooleanValue, FloatValue, IntValue, StringValue
from .definition import GraphQLScalarType
def parse_float_literal(ast):
    if isinstance(ast, (FloatValue, IntValue)):
        return float(ast.value)
    return None