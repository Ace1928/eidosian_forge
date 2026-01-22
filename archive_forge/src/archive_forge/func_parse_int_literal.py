from ..language.ast import BooleanValue, FloatValue, IntValue, StringValue
from .definition import GraphQLScalarType
def parse_int_literal(ast):
    if isinstance(ast, IntValue):
        num = int(ast.value)
        if MIN_INT <= num <= MAX_INT:
            return num