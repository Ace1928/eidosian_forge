from copy import copy, deepcopy
from typing import (
from ..error import GraphQLError
from ..language import ast, OperationType
from ..pyutils import inspect, is_collection, is_description
from .definition import (
from .directives import GraphQLDirective, specified_directives, is_directive
from .introspection import introspection_types
def remapped_type(type_: GraphQLType, type_map: TypeMap) -> GraphQLType:
    """Get a copy of the given type that uses this type map."""
    if is_wrapping_type(type_):
        type_ = cast(GraphQLWrappingType, type_)
        return type_.__class__(remapped_type(type_.of_type, type_map))
    type_ = cast(GraphQLNamedType, type_)
    return type_map.get(type_.name, type_)