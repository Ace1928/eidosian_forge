from enum import Enum
from typing import (
from ..error import GraphQLError
from ..language import (
from ..pyutils import (
from ..utilities.value_from_ast_untyped import value_from_ast_untyped
from .assert_name import assert_name, assert_enum_value_name
def assert_output_type(type_: Any) -> GraphQLOutputType:
    if not is_output_type(type_):
        raise TypeError(f'Expected {type_} to be a GraphQL output type.')
    return cast(GraphQLOutputType, type_)