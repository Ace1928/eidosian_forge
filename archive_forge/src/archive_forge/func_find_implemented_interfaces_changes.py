from enum import Enum
from typing import Any, Collection, Dict, List, NamedTuple, Union, cast
from ..language import print_ast
from ..pyutils import inspect, Undefined
from ..type import (
from ..utilities.sort_value_node import sort_value_node
from .ast_from_value import ast_from_value
def find_implemented_interfaces_changes(old_type: Union[GraphQLObjectType, GraphQLInterfaceType], new_type: Union[GraphQLObjectType, GraphQLInterfaceType]) -> List[Change]:
    schema_changes: List[Change] = []
    interfaces_diff = list_diff(old_type.interfaces, new_type.interfaces)
    for interface in interfaces_diff.added:
        schema_changes.append(DangerousChange(DangerousChangeType.IMPLEMENTED_INTERFACE_ADDED, f'{interface.name} added to interfaces implemented by {old_type.name}.'))
    for interface in interfaces_diff.removed:
        schema_changes.append(BreakingChange(BreakingChangeType.IMPLEMENTED_INTERFACE_REMOVED, f'{old_type.name} no longer implements interface {interface.name}.'))
    return schema_changes