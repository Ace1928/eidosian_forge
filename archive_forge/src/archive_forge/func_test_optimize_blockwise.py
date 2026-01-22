from __future__ import annotations
import pandas as pd
import pytest
import dask
import dask.dataframe as dd
def test_optimize_blockwise():
    from dask.array.optimization import optimize_blockwise
    df = pd.DataFrame({'x': range(10), 'y': range(10)})
    ddf = dd.from_pandas(df, npartitions=2)
    for _ in range(10):
        ddf['x'] = ddf.x + 1 + ddf.y
    graph = optimize_blockwise(ddf.dask)
    assert len(graph) <= 4