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
def order_by_limit(self, df: DataFrame, order_by: OrderBySpec, limit: int) -> DataFrame:
    assert_or_throw(not (len(order_by) > 0 and limit < 0), ValueError('for dask engine, limit is required by order by'))
    if len(order_by) == 0 and limit < 0:
        return df
    ndf = self.to_native(df)
    if len(order_by) == 0 or limit == 0:
        return self.to_df(ndf.head(limit, npartitions=-1, compute=False))
    ndf, sort_keys, asc = self._prepare_sort(ndf, order_by)

    def p_apply(df: Any) -> Any:
        return df.sort_values(sort_keys, ascending=asc).head(limit)
    meta = [(ndf[x].name, ndf[x].dtype) for x in ndf.columns]
    res = ndf.map_partitions(p_apply, meta=meta).compute()
    res = res.sort_values(sort_keys, ascending=asc).head(limit)
    return self.to_df(res[list(df.keys())])