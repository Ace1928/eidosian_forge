from enum import Enum
from typing import Dict, List, Sequence, Tuple, cast
import numpy as np
import pandas
from pandas._typing import IndexLabel
from pandas.api.types import is_scalar
def join_columns(left: pandas.Index, right: pandas.Index, left_on: IndexLabel, right_on: IndexLabel, suffixes: Tuple[str, str]) -> Tuple[pandas.Index, Dict[IndexLabel, IndexLabel], Dict[IndexLabel, IndexLabel]]:
    """
    Compute resulting columns for the two dataframes being merged.

    Parameters
    ----------
    left : pandas.Index
        Columns of the left frame to join.
    right : pandas.Index
        Columns of the right frame to join.
    left_on : list-like or scalar
        Column names on which the frames are joined in the left DataFrame.
    right_on : list-like or scalar
        Column names on which the frames are joined in the right DataFrame.
    suffixes : tuple[str, str]
        A 2-length sequence containing suffixes to append to the intersected columns.

    Returns
    -------
    pandas.Index, dict[IndexLabel -> IndexLabel], dict[IndexLabel -> IndexLabel]
        Returns columns for the resulting frame and mappings of old to new column
        names for `left` and `right` accordingly.

    Raises
    ------
    NotImplementedError
        Raised when one of the keys to join is an index level, pandas behaviour is really
        complicated in this case, so we're not supporting this case for now.
    """
    left_on = cast(Sequence[IndexLabel], [left_on] if is_scalar(left_on) else left_on)
    right_on = cast(Sequence[IndexLabel], [right_on] if is_scalar(right_on) else right_on)
    if len(left_on) == 1 and len(right_on) == 1 and (left_on[0] == right_on[0]):
        if left_on[0] not in left and right_on[0] not in right:
            left_on = []
            right_on = []
        elif left_on[0] not in left:
            left = left.insert(loc=0, item=left_on[0])
        elif right_on[0] not in right:
            right = right.insert(loc=0, item=right_on[0])
    if any((col not in left for col in left_on)) or any((col not in right for col in right_on)):
        raise NotImplementedError('Cases, where one of the keys to join is an index level, are not yet supported.')
    left_conflicts = set(left) & set(right) - set(right_on)
    right_conflicts = set(right) & set(left) - set(left_on)
    conflicting_cols = left_conflicts | right_conflicts

    def _get_new_name(col: IndexLabel, suffix: str) -> IndexLabel:
        if col in conflicting_cols:
            return (f'{col[0]}{suffix}', *col[1:]) if isinstance(col, tuple) else f'{col}{suffix}'
        else:
            return col
    left_renamer: Dict[IndexLabel, IndexLabel] = {}
    right_renamer: Dict[IndexLabel, IndexLabel] = {}
    new_left: List = []
    new_right: List = []
    for col in left:
        new_name = _get_new_name(col, suffixes[0])
        new_left.append(new_name)
        left_renamer[col] = new_name
    for col in right:
        if not (col in left_on and col in right_on):
            new_name = _get_new_name(col, suffixes[1])
            new_right.append(new_name)
            right_renamer[col] = new_name
    new_columns = pandas.Index(new_left + new_right)
    return (new_columns, left_renamer, right_renamer)