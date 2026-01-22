from enum import Enum
from typing import Any, Collection, Dict, List, NamedTuple, Union, cast
from ..language import print_ast
from ..pyutils import inspect, Undefined
from ..type import (
from ..utilities.sort_value_node import sort_value_node
from .ast_from_value import ast_from_value
def find_type_changes(old_schema: GraphQLSchema, new_schema: GraphQLSchema) -> List[Change]:
    schema_changes: List[Change] = []
    types_diff = dict_diff(old_schema.type_map, new_schema.type_map)
    for type_name, old_type in types_diff.removed.items():
        schema_changes.append(BreakingChange(BreakingChangeType.TYPE_REMOVED, f'Standard scalar {type_name} was removed because it is not referenced anymore.' if is_specified_scalar_type(old_type) else f'{type_name} was removed.'))
    for type_name, (old_type, new_type) in types_diff.persisted.items():
        if is_enum_type(old_type) and is_enum_type(new_type):
            schema_changes.extend(find_enum_type_changes(old_type, new_type))
        elif is_union_type(old_type) and is_union_type(new_type):
            schema_changes.extend(find_union_type_changes(old_type, new_type))
        elif is_input_object_type(old_type) and is_input_object_type(new_type):
            schema_changes.extend(find_input_object_type_changes(old_type, new_type))
        elif is_object_type(old_type) and is_object_type(new_type):
            schema_changes.extend(find_field_changes(old_type, new_type))
            schema_changes.extend(find_implemented_interfaces_changes(old_type, new_type))
        elif is_interface_type(old_type) and is_interface_type(new_type):
            schema_changes.extend(find_field_changes(old_type, new_type))
            schema_changes.extend(find_implemented_interfaces_changes(old_type, new_type))
        elif old_type.__class__ is not new_type.__class__:
            schema_changes.append(BreakingChange(BreakingChangeType.TYPE_CHANGED_KIND, f'{type_name} changed from {type_kind_name(old_type)} to {type_kind_name(new_type)}.'))
    return schema_changes