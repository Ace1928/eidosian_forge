from __future__ import annotations
from datetime import datetime
import numpy as np
import pandas as pd
import pytest
import dask
import dask.array as da
import dask.dataframe as dd
from dask import config
from dask.blockwise import Blockwise
from dask.dataframe._compat import PANDAS_GE_200, tm
from dask.dataframe.io.io import _meta_from_array
from dask.dataframe.optimize import optimize
from dask.dataframe.utils import assert_eq, get_string_dtype, pyarrow_strings_enabled
from dask.delayed import Delayed, delayed
from dask.utils_test import hlg_layer_topological
def test_from_delayed():
    df = pd.DataFrame(data=np.random.normal(size=(10, 4)), columns=list('abcd'))
    parts = [df.iloc[:1], df.iloc[1:3], df.iloc[3:6], df.iloc[6:10]]
    dfs = [delayed(parts.__getitem__)(i) for i in range(4)]
    meta = dfs[0].compute()
    my_len = lambda x: pd.Series([len(x)])
    for divisions in [None, [0, 1, 3, 6, 10]]:
        ddf = dd.from_delayed(dfs, meta=meta, divisions=divisions)
        assert_eq(ddf, df)
        assert list(ddf.map_partitions(my_len).compute()) == [1, 2, 3, 4]
        assert ddf.known_divisions == (divisions is not None)
        s = dd.from_delayed([d.a for d in dfs], meta=meta.a, divisions=divisions)
        assert_eq(s, df.a)
        assert list(s.map_partitions(my_len).compute()) == [1, 2, 3, 4]
        assert ddf.known_divisions == (divisions is not None)
    meta2 = [(c, 'f8') for c in df.columns]
    check_ddf = dd.from_delayed(dfs, meta=meta2)
    if not DASK_EXPR_ENABLED:
        assert isinstance(check_ddf.dask.layers[check_ddf._name], Blockwise)
    assert_eq(dd.from_delayed(dfs, meta=meta2), df)
    assert_eq(dd.from_delayed([d.a for d in dfs], meta=('a', 'f8')), df.a)
    with pytest.raises(ValueError):
        dd.from_delayed(dfs, meta=meta, divisions=[0, 1, 3, 6])
    with pytest.raises(ValueError) as e:
        dd.from_delayed(dfs, meta=meta.a).compute()
    assert str(e.value).startswith('Metadata mismatch found in `from_delayed`')