from itertools import chain
from typing import Any, Dict, List, Optional, Tuple, Union, cast
from ...error import GraphQLError
from ...language import (
from ...type import (
from ...utilities import type_from_ast
from ...utilities.sort_value_node import sort_value_node
from . import ValidationContext, ValidationRule
def collect_fields_and_fragment_names(context: ValidationContext, parent_type: Optional[GraphQLNamedType], selection_set: SelectionSetNode, node_and_defs: NodeAndDefCollection, fragment_names: Dict[str, bool]) -> None:
    for selection in selection_set.selections:
        if isinstance(selection, FieldNode):
            field_name = selection.name.value
            field_def = parent_type.fields.get(field_name) if is_object_type(parent_type) or is_interface_type(parent_type) else None
            response_name = selection.alias.value if selection.alias else field_name
            if not node_and_defs.get(response_name):
                node_and_defs[response_name] = []
            node_and_defs[response_name].append(cast(NodeAndDef, (parent_type, selection, field_def)))
        elif isinstance(selection, FragmentSpreadNode):
            fragment_names[selection.name.value] = True
        elif isinstance(selection, InlineFragmentNode):
            type_condition = selection.type_condition
            inline_fragment_type = type_from_ast(context.schema, type_condition) if type_condition else parent_type
            collect_fields_and_fragment_names(context, inline_fragment_type, selection.selection_set, node_and_defs, fragment_names)