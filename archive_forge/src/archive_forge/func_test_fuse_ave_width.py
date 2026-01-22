from __future__ import annotations
import pandas as pd
import pytest
import dask
import dask.dataframe as dd
def test_fuse_ave_width():
    df = pd.DataFrame({'x': range(10)})
    df = dd.from_pandas(df, npartitions=5)
    s = df.x + 1 + (df.x + 2)
    with dask.config.set({'optimization.fuse.ave-width': 4}):
        a = s.__dask_optimize__(s.dask, s.__dask_keys__())
    b = s.__dask_optimize__(s.dask, s.__dask_keys__())
    assert len(a) <= 15
    assert len(b) <= 15