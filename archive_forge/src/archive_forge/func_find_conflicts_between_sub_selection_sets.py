from itertools import chain
from typing import Any, Dict, List, Optional, Tuple, Union, cast
from ...error import GraphQLError
from ...language import (
from ...type import (
from ...utilities import type_from_ast
from ...utilities.sort_value_node import sort_value_node
from . import ValidationContext, ValidationRule
def find_conflicts_between_sub_selection_sets(context: ValidationContext, cached_fields_and_fragment_names: Dict, compared_fragment_pairs: 'PairSet', are_mutually_exclusive: bool, parent_type1: Optional[GraphQLNamedType], selection_set1: SelectionSetNode, parent_type2: Optional[GraphQLNamedType], selection_set2: SelectionSetNode) -> List[Conflict]:
    """Find conflicts between sub selection sets.

    Find all conflicts found between two selection sets, including those found via
    spreading in fragments. Called when determining if conflicts exist between the
    sub-fields of two overlapping fields.
    """
    conflicts: List[Conflict] = []
    field_map1, fragment_names1 = get_fields_and_fragment_names(context, cached_fields_and_fragment_names, parent_type1, selection_set1)
    field_map2, fragment_names2 = get_fields_and_fragment_names(context, cached_fields_and_fragment_names, parent_type2, selection_set2)
    collect_conflicts_between(context, conflicts, cached_fields_and_fragment_names, compared_fragment_pairs, are_mutually_exclusive, field_map1, field_map2)
    if fragment_names2:
        for fragment_name2 in fragment_names2:
            collect_conflicts_between_fields_and_fragment(context, conflicts, cached_fields_and_fragment_names, compared_fragment_pairs, are_mutually_exclusive, field_map1, fragment_name2)
    if fragment_names1:
        for fragment_name1 in fragment_names1:
            collect_conflicts_between_fields_and_fragment(context, conflicts, cached_fields_and_fragment_names, compared_fragment_pairs, are_mutually_exclusive, field_map2, fragment_name1)
    for fragment_name1 in fragment_names1:
        for fragment_name2 in fragment_names2:
            collect_conflicts_between_fragments(context, conflicts, cached_fields_and_fragment_names, compared_fragment_pairs, are_mutually_exclusive, fragment_name1, fragment_name2)
    return conflicts