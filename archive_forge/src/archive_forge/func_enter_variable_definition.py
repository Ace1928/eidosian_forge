from typing import Any, Dict, Optional, cast
from ...error import GraphQLError
from ...language import (
from ...pyutils import Undefined
from ...type import GraphQLNonNull, GraphQLSchema, GraphQLType, is_non_null_type
from ...utilities import type_from_ast, is_type_sub_type_of
from . import ValidationContext, ValidationRule
def enter_variable_definition(self, node: VariableDefinitionNode, *_args: Any) -> None:
    self.var_def_map[node.variable.name.value] = node