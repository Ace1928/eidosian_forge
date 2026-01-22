from typing import Any, Callable, Dict, List, Tuple
import dask.array as np
import dask.dataframe as pd
import numpy
import pandas
from triad.utils.assertion import assert_or_throw
from triad.utils.pandas_like import PandasLikeUtils
from qpd import QPDEngine, run_sql
from qpd.dataframe import Column, DataFrame
from qpd.specs import (
from qpd_pandas.engine import _RowsIndexer
def gp_apply(gdf: Any) -> Any:
    if len(sort_keys) > 0:
        gdf = gdf.sort_values(sort_keys, ascending=asc)
    if not func.has_windowframe:
        res: Any = agg_func(gdf[col])
    else:
        wf = func.window.windowframe
        assert_or_throw(isinstance(wf, RollingWindowFrame), NotImplementedError(f'{wf}'))
        min_p = 0 if func.name in ['count', 'min', 'max'] else 1
        if wf.start is None and wf.end == 0:
            rolling_func: Callable = lambda x: x.expanding(min_periods=min_p)
        elif wf.window is not None and wf.end == 0:
            rolling_func: Callable = lambda x: x.rolling(window=wf.window, min_periods=min_p)
        else:
            rolling_func: Callable = lambda x: x.rolling(window=_RowsIndexer(wf.start, wf.end), min_periods=min_p)
        res = agg_func(rolling_func(gdf[col]))
    gdf[dest_col_name] = res
    return gdf.reset_index(drop=True)