from enum import Enum
from typing import (
from ..error import GraphQLError
from ..language import (
from ..pyutils import (
from ..utilities.value_from_ast_untyped import value_from_ast_untyped
from .assert_name import assert_name, assert_enum_value_name
def assert_leaf_type(type_: Any) -> GraphQLLeafType:
    if not is_leaf_type(type_):
        raise TypeError(f'Expected {type_} to be a GraphQL leaf type.')
    return cast(GraphQLLeafType, type_)