from typing import List, Optional, Tuple, Dict, Iterable, overload, Union
from altair import (
from altair.utils._vegafusion_data import get_inline_tables, import_vegafusion
from altair.utils.core import DataFrameLike
from altair.utils.schemapi import Undefined
def get_group_mark_for_scope(vega_spec: dict, scope: Scope) -> Optional[dict]:
    """Get the group mark at a particular scope

    Parameters
    ----------
    vega_spec : dict
        Top-level Vega specification dictionary
    scope : tuple of int
        Scope tuple. If empty, the original Vega specification is returned.
        Otherwise, the nested group mark at the scope specified is returned.

    Returns
    -------
    dict or None
        Top-level Vega spec (if scope is empty)
        or group mark (if scope is non-empty)
        or None (if group mark at scope does not exist)

    Examples
    --------
    >>> spec = {
    ...     "marks": [
    ...         {
    ...             "type": "group",
    ...             "marks": [{"type": "symbol"}]
    ...         },
    ...         {
    ...             "type": "group",
    ...             "marks": [{"type": "rect"}]}
    ...     ]
    ... }
    >>> get_group_mark_for_scope(spec, (1,))
    {'type': 'group', 'marks': [{'type': 'rect'}]}
    """
    group = vega_spec
    for scope_value in scope:
        group_index = 0
        child_group = None
        for mark in group.get('marks', []):
            if mark.get('type') == 'group':
                if group_index == scope_value:
                    child_group = mark
                    break
                group_index += 1
        if child_group is None:
            return None
        group = child_group
    return group