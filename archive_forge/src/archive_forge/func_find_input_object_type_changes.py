from enum import Enum
from typing import Any, Collection, Dict, List, NamedTuple, Union, cast
from ..language import print_ast
from ..pyutils import inspect, Undefined
from ..type import (
from ..utilities.sort_value_node import sort_value_node
from .ast_from_value import ast_from_value
def find_input_object_type_changes(old_type: Union[GraphQLObjectType, GraphQLInterfaceType], new_type: Union[GraphQLObjectType, GraphQLInterfaceType]) -> List[Change]:
    schema_changes: List[Change] = []
    fields_diff = dict_diff(old_type.fields, new_type.fields)
    for field_name, new_field in fields_diff.added.items():
        if is_required_input_field(new_field):
            schema_changes.append(BreakingChange(BreakingChangeType.REQUIRED_INPUT_FIELD_ADDED, f'A required field {field_name} on input type {old_type.name} was added.'))
        else:
            schema_changes.append(DangerousChange(DangerousChangeType.OPTIONAL_INPUT_FIELD_ADDED, f'An optional field {field_name} on input type {old_type.name} was added.'))
    for field_name in fields_diff.removed:
        schema_changes.append(BreakingChange(BreakingChangeType.FIELD_REMOVED, f'{old_type.name}.{field_name} was removed.'))
    for field_name, (old_field, new_field) in fields_diff.persisted.items():
        is_safe = is_change_safe_for_input_object_field_or_field_arg(old_field.type, new_field.type)
        if not is_safe:
            schema_changes.append(BreakingChange(BreakingChangeType.FIELD_CHANGED_KIND, f'{old_type.name}.{field_name} changed type from {old_field.type} to {new_field.type}.'))
    return schema_changes