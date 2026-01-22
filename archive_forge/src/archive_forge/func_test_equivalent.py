from __future__ import annotations
from collections.abc import Hashable
import numpy as np
import pandas as pd
import pytest
from xarray.core import duck_array_ops, utils
from xarray.core.utils import either_dict_or_kwargs, infix_dims, iterate_nested
from xarray.tests import assert_array_equal, requires_dask
def test_equivalent(self):
    assert utils.equivalent(0, 0)
    assert utils.equivalent(np.nan, np.nan)
    assert utils.equivalent(0, np.array(0.0))
    assert utils.equivalent([0], np.array([0]))
    assert utils.equivalent(np.array([0]), [0])
    assert utils.equivalent(np.arange(3), 1.0 * np.arange(3))
    assert not utils.equivalent(0, np.zeros(3))