from enum import Enum
from typing import (
from ..error import GraphQLError
from ..language import (
from ..pyutils import (
from ..utilities.value_from_ast_untyped import value_from_ast_untyped
from .assert_name import assert_name, assert_enum_value_name
def resolve_thunk(thunk: Thunk[T]) -> T:
    """Resolve the given thunk.

    Used while defining GraphQL types to allow for circular references in otherwise
    immutable type definitions.
    """
    return thunk() if callable(thunk) else thunk