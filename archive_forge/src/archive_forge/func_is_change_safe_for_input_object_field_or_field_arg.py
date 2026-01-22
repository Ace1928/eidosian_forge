from enum import Enum
from typing import Any, Collection, Dict, List, NamedTuple, Union, cast
from ..language import print_ast
from ..pyutils import inspect, Undefined
from ..type import (
from ..utilities.sort_value_node import sort_value_node
from .ast_from_value import ast_from_value
def is_change_safe_for_input_object_field_or_field_arg(old_type: GraphQLType, new_type: GraphQLType) -> bool:
    if is_list_type(old_type):
        return is_list_type(new_type) and is_change_safe_for_input_object_field_or_field_arg(cast(GraphQLList, old_type).of_type, cast(GraphQLList, new_type).of_type)
    if is_non_null_type(old_type):
        return is_non_null_type(new_type) and is_change_safe_for_input_object_field_or_field_arg(cast(GraphQLNonNull, old_type).of_type, cast(GraphQLNonNull, new_type).of_type) or (not is_non_null_type(new_type) and is_change_safe_for_input_object_field_or_field_arg(cast(GraphQLNonNull, old_type).of_type, new_type))
    return is_named_type(new_type) and cast(GraphQLNamedType, old_type).name == cast(GraphQLNamedType, new_type).name