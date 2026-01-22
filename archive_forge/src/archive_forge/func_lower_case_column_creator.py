from __future__ import annotations
import os
import sys
from typing import Any, TypeVar, Callable, Optional, NamedTuple
from typing_extensions import TypeAlias
from .._extras import pandas as pd
def lower_case_column_creator(df: pd.DataFrame) -> pd.DataFrame:
    return lower_case_column(df, necessary_column)