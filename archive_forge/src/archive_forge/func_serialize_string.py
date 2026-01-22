from math import isfinite
from typing import Any, Mapping
from ..error import GraphQLError
from ..pyutils import inspect
from ..language.ast import (
from ..language.printer import print_ast
from .definition import GraphQLNamedType, GraphQLScalarType
def serialize_string(output_value: Any) -> str:
    if isinstance(output_value, str):
        return output_value
    if isinstance(output_value, bool):
        return 'true' if output_value else 'false'
    if isinstance(output_value, int) or (isinstance(output_value, float) and isfinite(output_value)):
        return str(output_value)
    if type(output_value).__module__ == 'builtins':
        raise GraphQLError('String cannot represent value: ' + inspect(output_value))
    return str(output_value)