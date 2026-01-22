from enum import Enum
from typing import Any, Collection, Dict, List, NamedTuple, Union, cast
from ..language import print_ast
from ..pyutils import inspect, Undefined
from ..type import (
from ..utilities.sort_value_node import sort_value_node
from .ast_from_value import ast_from_value
def find_schema_changes(old_schema: GraphQLSchema, new_schema: GraphQLSchema) -> List[Change]:
    return find_type_changes(old_schema, new_schema) + find_directive_changes(old_schema, new_schema)