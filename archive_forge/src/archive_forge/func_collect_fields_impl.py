from typing import Any, Dict, List, Set, Union, cast
from ..language import (
from ..type import (
from ..utilities.type_from_ast import type_from_ast
from .values import get_directive_values
def collect_fields_impl(schema: GraphQLSchema, fragments: Dict[str, FragmentDefinitionNode], variable_values: Dict[str, Any], runtime_type: GraphQLObjectType, selection_set: SelectionSetNode, fields: Dict[str, List[FieldNode]], visited_fragment_names: Set[str]) -> None:
    """Collect fields (internal implementation)."""
    for selection in selection_set.selections:
        if isinstance(selection, FieldNode):
            if not should_include_node(variable_values, selection):
                continue
            name = get_field_entry_key(selection)
            fields.setdefault(name, []).append(selection)
        elif isinstance(selection, InlineFragmentNode):
            if not should_include_node(variable_values, selection) or not does_fragment_condition_match(schema, selection, runtime_type):
                continue
            collect_fields_impl(schema, fragments, variable_values, runtime_type, selection.selection_set, fields, visited_fragment_names)
        elif isinstance(selection, FragmentSpreadNode):
            frag_name = selection.name.value
            if frag_name in visited_fragment_names or not should_include_node(variable_values, selection):
                continue
            visited_fragment_names.add(frag_name)
            fragment = fragments.get(frag_name)
            if not fragment or not does_fragment_condition_match(schema, fragment, runtime_type):
                continue
            collect_fields_impl(schema, fragments, variable_values, runtime_type, fragment.selection_set, fields, visited_fragment_names)