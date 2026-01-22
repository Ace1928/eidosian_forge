from __future__ import annotations
import os
import sys
from typing import Any, TypeVar, Callable, Optional, NamedTuple
from typing_extensions import TypeAlias
from .._extras import pandas as pd
def duplicated_rows_validator(df: pd.DataFrame, fields: list[str]=['prompt', 'completion']) -> Remediation:
    """
    This validator will suggest to the user to remove duplicate rows if they exist.
    """
    duplicated_rows = df.duplicated(subset=fields)
    duplicated_indexes = df.reset_index().index[duplicated_rows].tolist()
    immediate_msg = None
    optional_msg = None
    optional_fn = None
    if len(duplicated_indexes) > 0:
        immediate_msg = f'\n- There are {len(duplicated_indexes)} duplicated {'-'.join(fields)} sets. These are rows: {duplicated_indexes}'
        optional_msg = f'Remove {len(duplicated_indexes)} duplicate rows'

        def optional_fn(x: Any) -> Any:
            return x.drop_duplicates(subset=fields)
    return Remediation(name='duplicated_rows', immediate_msg=immediate_msg, optional_msg=optional_msg, optional_fn=optional_fn)