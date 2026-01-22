from __future__ import annotations
import pickle
import re
import sys
import warnings
from collections.abc import Hashable
from copy import deepcopy
from textwrap import dedent
from typing import Any, Final, Literal, cast
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray import (
from xarray.coding.times import CFDatetimeCoder
from xarray.core import dtypes
from xarray.core.common import full_like
from xarray.core.coordinates import Coordinates
from xarray.core.indexes import Index, PandasIndex, filter_indexes_from_coords
from xarray.core.types import QueryEngineOptions, QueryParserOptions
from xarray.core.utils import is_scalar
from xarray.testing import _assert_internal_invariants
from xarray.tests import (
@requires_scipy
@pytest.mark.parametrize('use_dask', [True, False])
def test_curvefit(self, use_dask) -> None:
    if use_dask and (not has_dask):
        pytest.skip('requires dask')

    def exp_decay(t, n0, tau=1):
        return n0 * np.exp(-t / tau)
    t = np.arange(0, 5, 0.5)
    da = DataArray(np.stack([exp_decay(t, 3, 3), exp_decay(t, 5, 4), np.nan * t], axis=-1), dims=('t', 'x'), coords={'t': t, 'x': [0, 1, 2]})
    da[0, 0] = np.nan
    expected = DataArray([[3, 3], [5, 4], [np.nan, np.nan]], dims=('x', 'param'), coords={'x': [0, 1, 2], 'param': ['n0', 'tau']})
    if use_dask:
        da = da.chunk({'x': 1})
    fit = da.curvefit(coords=[da.t], func=exp_decay, p0={'n0': 4}, bounds={'tau': (2, 6)})
    assert_allclose(fit.curvefit_coefficients, expected, rtol=0.001)
    da = da.compute()
    fit = da.curvefit(coords='t', func=np.power, reduce_dims='x', param_names=['a'])
    assert 'a' in fit.param
    assert 'x' not in fit.dims