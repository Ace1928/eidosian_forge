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
def handle_field_error(self, error: GraphQLError, return_type: GraphQLOutputType) -> None:
    if is_non_null_type(return_type):
        raise error
    self.errors.append(error)
    return None