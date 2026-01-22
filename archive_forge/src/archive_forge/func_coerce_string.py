from ..language.ast import BooleanValue, FloatValue, IntValue, StringValue
from .definition import GraphQLScalarType
def coerce_string(value):
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return u'true' if value else u'false'
    return str(value)