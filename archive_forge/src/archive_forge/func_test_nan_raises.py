from __future__ import annotations
import warnings
import pytest
from packaging.version import parse as parse_version
import numpy as np
import dask.array as da
import dask.array.stats
from dask.array.utils import allclose, assert_eq
from dask.delayed import Delayed
@pytest.mark.parametrize('func, nargs', [(dask.array.stats.ttest_1samp, 2), (dask.array.stats.ttest_rel, 2), (dask.array.stats.skewtest, 1), (dask.array.stats.kurtosis, 1), (dask.array.stats.kurtosistest, 1), (dask.array.stats.normaltest, 1), (dask.array.stats.moment, 1)])
@pytest.mark.parametrize('nan_policy', ['omit', 'raise'])
def test_nan_raises(func, nargs, nan_policy):
    with pytest.raises(NotImplementedError):
        func(*(None,) * nargs, nan_policy=nan_policy)