from enum import Enum
from typing import Any, Collection, Dict, List, NamedTuple, Union, cast
from ..language import print_ast
from ..pyutils import inspect, Undefined
from ..type import (
from ..utilities.sort_value_node import sort_value_node
from .ast_from_value import ast_from_value
def find_union_type_changes(old_type: GraphQLUnionType, new_type: GraphQLUnionType) -> List[Change]:
    schema_changes: List[Change] = []
    possible_types_diff = list_diff(old_type.types, new_type.types)
    for possible_type in possible_types_diff.added:
        schema_changes.append(DangerousChange(DangerousChangeType.TYPE_ADDED_TO_UNION, f'{possible_type.name} was added to union type {old_type.name}.'))
    for possible_type in possible_types_diff.removed:
        schema_changes.append(BreakingChange(BreakingChangeType.TYPE_REMOVED_FROM_UNION, f'{possible_type.name} was removed from union type {old_type.name}.'))
    return schema_changes