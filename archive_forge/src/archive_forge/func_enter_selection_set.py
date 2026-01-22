from itertools import chain
from typing import Any, Dict, List, Optional, Tuple, Union, cast
from ...error import GraphQLError
from ...language import (
from ...type import (
from ...utilities import type_from_ast
from ...utilities.sort_value_node import sort_value_node
from . import ValidationContext, ValidationRule
def enter_selection_set(self, selection_set: SelectionSetNode, *_args: Any) -> None:
    conflicts = find_conflicts_within_selection_set(self.context, self.cached_fields_and_fragment_names, self.compared_fragment_pairs, self.context.get_parent_type(), selection_set)
    for (reason_name, reason), fields1, fields2 in conflicts:
        reason_msg = reason_message(reason)
        self.report_error(GraphQLError(f"Fields '{reason_name}' conflict because {reason_msg}. Use different aliases on the fields to fetch both if this was intentional.", fields1 + fields2))