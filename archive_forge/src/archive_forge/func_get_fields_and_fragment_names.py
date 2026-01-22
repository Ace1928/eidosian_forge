from itertools import chain
from typing import Any, Dict, List, Optional, Tuple, Union, cast
from ...error import GraphQLError
from ...language import (
from ...type import (
from ...utilities import type_from_ast
from ...utilities.sort_value_node import sort_value_node
from . import ValidationContext, ValidationRule
def get_fields_and_fragment_names(context: ValidationContext, cached_fields_and_fragment_names: Dict, parent_type: Optional[GraphQLNamedType], selection_set: SelectionSetNode) -> Tuple[NodeAndDefCollection, List[str]]:
    """Get fields and referenced fragment names

    Given a selection set, return the collection of fields (a mapping of response name
    to field nodes and definitions) as well as a list of fragment names referenced via
    fragment spreads.
    """
    cached = cached_fields_and_fragment_names.get(selection_set)
    if not cached:
        node_and_defs: NodeAndDefCollection = {}
        fragment_names: Dict[str, bool] = {}
        collect_fields_and_fragment_names(context, parent_type, selection_set, node_and_defs, fragment_names)
        cached = (node_and_defs, list(fragment_names))
        cached_fields_and_fragment_names[selection_set] = cached
    return cached