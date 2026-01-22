from ..language.ast import BooleanValue, FloatValue, IntValue, StringValue
from .definition import GraphQLScalarType
def parse_id_literal(ast):
    if isinstance(ast, (StringValue, IntValue)):
        return ast.value
    return None