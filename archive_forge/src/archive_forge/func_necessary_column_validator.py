from __future__ import annotations
import os
import sys
from typing import Any, TypeVar, Callable, Optional, NamedTuple
from typing_extensions import TypeAlias
from .._extras import pandas as pd
def necessary_column_validator(df: pd.DataFrame, necessary_column: str) -> Remediation:
    """
    This validator will ensure that the necessary column is present in the dataframe.
    """

    def lower_case_column(df: pd.DataFrame, column: Any) -> pd.DataFrame:
        cols = [c for c in df.columns if str(c).lower() == column]
        df.rename(columns={cols[0]: column.lower()}, inplace=True)
        return df
    immediate_msg = None
    necessary_fn = None
    necessary_msg = None
    error_msg = None
    if necessary_column not in df.columns:
        if necessary_column in [str(c).lower() for c in df.columns]:

            def lower_case_column_creator(df: pd.DataFrame) -> pd.DataFrame:
                return lower_case_column(df, necessary_column)
            necessary_fn = lower_case_column_creator
            immediate_msg = f'\n- The `{necessary_column}` column/key should be lowercase'
            necessary_msg = f'Lower case column name to `{necessary_column}`'
        else:
            error_msg = f'`{necessary_column}` column/key is missing. Please make sure you name your columns/keys appropriately, then retry'
    return Remediation(name='necessary_column', immediate_msg=immediate_msg, necessary_msg=necessary_msg, necessary_fn=necessary_fn, error_msg=error_msg)