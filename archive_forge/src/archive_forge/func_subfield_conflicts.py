from itertools import chain
from typing import Any, Dict, List, Optional, Tuple, Union, cast
from ...error import GraphQLError
from ...language import (
from ...type import (
from ...utilities import type_from_ast
from ...utilities.sort_value_node import sort_value_node
from . import ValidationContext, ValidationRule
def subfield_conflicts(conflicts: List[Conflict], response_name: str, node1: FieldNode, node2: FieldNode) -> Optional[Conflict]:
    """Check whether there are conflicts between sub-fields.

    Given a series of Conflicts which occurred between two sub-fields, generate a single
    Conflict.
    """
    if conflicts:
        return ((response_name, [conflict[0] for conflict in conflicts]), list(chain([node1], *[conflict[1] for conflict in conflicts])), list(chain([node2], *[conflict[2] for conflict in conflicts])))
    return None