from math import isfinite
from typing import Any, Mapping
from ..error import GraphQLError
from ..pyutils import inspect
from ..language.ast import (
from ..language.printer import print_ast
from .definition import GraphQLNamedType, GraphQLScalarType
def is_specified_scalar_type(type_: GraphQLNamedType) -> bool:
    """Check whether the given named GraphQL type is a specified scalar type."""
    return type_.name in specified_scalar_types