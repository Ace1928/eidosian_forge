from asyncio import ensure_future, gather
from collections.abc import Mapping
from inspect import isawaitable
from typing import (
from ..error import GraphQLError, GraphQLFormattedError, located_error
from ..language import (
from ..pyutils import (
from ..type import (
from .collect_fields import collect_fields, collect_sub_fields
from .middleware import MiddlewareManager
from .values import get_argument_values, get_variable_values
def invalid_return_type_error(return_type: GraphQLObjectType, result: Any, field_nodes: List[FieldNode]) -> GraphQLError:
    """Create a GraphQLError for an invalid return type."""
    return GraphQLError(f"Expected value of type '{return_type.name}' but got: {inspect(result)}.", field_nodes)