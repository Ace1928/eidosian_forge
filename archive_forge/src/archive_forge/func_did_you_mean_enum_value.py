from enum import Enum
from typing import (
from ..error import GraphQLError
from ..language import (
from ..pyutils import (
from ..utilities.value_from_ast_untyped import value_from_ast_untyped
from .assert_name import assert_name, assert_enum_value_name
def did_you_mean_enum_value(enum_type: GraphQLEnumType, unknown_value_str: str) -> str:
    suggested_values = suggestion_list(unknown_value_str, enum_type.values)
    return did_you_mean(suggested_values, 'the enum value')