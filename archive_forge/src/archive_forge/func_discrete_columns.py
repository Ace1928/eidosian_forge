from __future__ import annotations
import typing
from copy import copy, deepcopy
from typing import Iterable, List, cast, overload
import pandas as pd
from ._utils import array_kind, check_required_aesthetics, ninteraction
from .exceptions import PlotnineError
from .mapping.aes import NO_GROUP, SCALED_AESTHETICS, aes
from .mapping.evaluation import evaluate, stage
def discrete_columns(df: pd.DataFrame, ignore: Sequence[str] | pd.Index) -> list[str]:
    """
    Return a list of the discrete columns in the dataframe

    Parameters
    ----------
    df : dataframe
        Data
    ignore : list[str]
        A list|set|tuple with the names of the columns to skip.
    """
    lst = []
    for col in df:
        if array_kind.discrete(df[col]) and col not in ignore:
            try:
                hash(df[col].iloc[0])
            except TypeError:
                continue
            lst.append(str(col))
    return lst