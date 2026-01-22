from __future__ import annotations
import itertools
import types
import typing
from copy import copy, deepcopy
import numpy as np
import pandas as pd
import pandas.api.types as pdtypes
from .._utils import cross_join, match
from ..exceptions import PlotnineError
from ..scales.scales import Scales
from .strips import Strips
def unique_combs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate all possible combinations of the values in the columns
    """

    def _unique(s: pd.Series[Any]) -> npt.NDArray[Any] | pd.Index:
        if isinstance(s.dtype, pdtypes.CategoricalDtype):
            return s.cat.categories
        return s.unique()
    lst = (_unique(x) for _, x in df.items())
    rows = list(itertools.product(*lst))
    _df = pd.DataFrame(rows, columns=df.columns)
    for col in df:
        t = df[col].dtype
        _df[col] = _df[col].astype(t, copy=False)
    return _df