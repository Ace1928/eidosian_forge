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
def test_curvefit_helpers(self) -> None:

    def exp_decay(t, n0, tau=1):
        return n0 * np.exp(-t / tau)
    params, func_args = xr.core.dataset._get_func_args(exp_decay, [])
    assert params == ['n0', 'tau']
    param_defaults, bounds_defaults = xr.core.dataset._initialize_curvefit_params(params, {'n0': 4}, {'tau': [5, np.inf]}, func_args)
    assert param_defaults == {'n0': 4, 'tau': 6}
    assert bounds_defaults == {'n0': (-np.inf, np.inf), 'tau': (5, np.inf)}
    param_defaults, bounds_defaults = xr.core.dataset._initialize_curvefit_params(params=params, p0={'n0': 4}, bounds={'tau': [DataArray([3, 4], coords=[('x', [1, 2])]), np.inf]}, func_args=func_args)
    assert param_defaults['n0'] == 4
    assert (param_defaults['tau'] == xr.DataArray([4, 5], coords=[('x', [1, 2])])).all()
    assert bounds_defaults['n0'] == (-np.inf, np.inf)
    assert (bounds_defaults['tau'][0] == DataArray([3, 4], coords=[('x', [1, 2])])).all()
    assert bounds_defaults['tau'][1] == np.inf
    param_names = ['a']
    params, func_args = xr.core.dataset._get_func_args(np.power, param_names)
    assert params == param_names
    with pytest.raises(ValueError):
        xr.core.dataset._get_func_args(np.power, [])