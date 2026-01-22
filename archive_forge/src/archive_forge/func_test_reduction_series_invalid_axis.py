from __future__ import annotations
import contextlib
import warnings
from datetime import datetime
import numpy as np
import pandas as pd
import pytest
from pandas.api.types import is_scalar
import dask.dataframe as dd
from dask.array.numpy_compat import NUMPY_GE_125
from dask.dataframe._compat import (
from dask.dataframe.utils import (
def test_reduction_series_invalid_axis():
    dsk = {('x', 0): pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]}, index=[0, 1, 3]), ('x', 1): pd.DataFrame({'a': [4, 5, 6], 'b': [3, 2, 1]}, index=[5, 6, 8]), ('x', 2): pd.DataFrame({'a': [7, 8, 9], 'b': [0, 0, 0]}, index=[9, 9, 9])}
    meta = make_meta({'a': 'i8', 'b': 'i8'}, index=pd.Index([], 'i8'), parent_meta=pd.DataFrame())
    if DASK_EXPR_ENABLED:
        ddf1 = dd.repartition(pd.concat(dsk.values()), [0, 4, 9, 9])
    else:
        ddf1 = dd.DataFrame(dsk, 'x', meta, [0, 4, 9, 9])
    pdf1 = ddf1.compute()
    for axis in [1, 'columns']:
        for s in [ddf1.a, pdf1.a]:
            pytest.raises(ValueError, lambda s=s, axis=axis: s.sum(axis=axis))
            pytest.raises(ValueError, lambda s=s, axis=axis: s.prod(axis=axis))
            pytest.raises(ValueError, lambda s=s, axis=axis: s.product(axis=axis))
            pytest.raises(ValueError, lambda s=s, axis=axis: s.min(axis=axis))
            pytest.raises(ValueError, lambda s=s, axis=axis: s.max(axis=axis))
            pytest.raises((TypeError, ValueError), lambda s=s, axis=axis: s.count(axis=axis))
            pytest.raises(ValueError, lambda s=s, axis=axis: s.std(axis=axis))
            pytest.raises(ValueError, lambda s=s, axis=axis: s.var(axis=axis))
            pytest.raises(ValueError, lambda s=s, axis=axis: s.sem(axis=axis))
            pytest.raises(ValueError, lambda s=s, axis=axis: s.mean(axis=axis))