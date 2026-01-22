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
@pytest.mark.parametrize('split_every', [False, 2])
def test_deterministic_reduction_names(split_every):
    df = pd.DataFrame({'x': [1, 2, 3, 4], 'y': [5, 6, 7, 8]})
    ddf = dd.from_pandas(df, npartitions=2)
    for x in [ddf, ddf.x]:
        assert x.sum(split_every=split_every)._name == x.sum(split_every=split_every)._name
        assert x.prod(split_every=split_every)._name == x.prod(split_every=split_every)._name
        assert x.product(split_every=split_every)._name == x.product(split_every=split_every)._name
        assert x.min(split_every=split_every)._name == x.min(split_every=split_every)._name
        assert x.max(split_every=split_every)._name == x.max(split_every=split_every)._name
        assert x.count(split_every=split_every)._name == x.count(split_every=split_every)._name
        assert x.std(split_every=split_every)._name == x.std(split_every=split_every)._name
        assert x.var(split_every=split_every)._name == x.var(split_every=split_every)._name
        assert x.sem(split_every=split_every)._name == x.sem(split_every=split_every)._name
        assert x.mean(split_every=split_every)._name == x.mean(split_every=split_every)._name
    assert ddf.x.nunique(split_every=split_every)._name == ddf.x.nunique(split_every=split_every)._name