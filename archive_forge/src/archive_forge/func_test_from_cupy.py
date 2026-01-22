from __future__ import annotations
import warnings
from abc import ABC
from copy import copy, deepcopy
from datetime import datetime, timedelta
from textwrap import dedent
from typing import Generic
import numpy as np
import pandas as pd
import pytest
import pytz
from xarray import DataArray, Dataset, IndexVariable, Variable, set_options
from xarray.core import dtypes, duck_array_ops, indexing
from xarray.core.common import full_like, ones_like, zeros_like
from xarray.core.indexing import (
from xarray.core.types import T_DuckArray
from xarray.core.utils import NDArrayMixin
from xarray.core.variable import as_compatible_data, as_variable
from xarray.namedarray.pycompat import array_type
from xarray.tests import (
from xarray.tests.test_namedarray import NamedArraySubclassobjects
@requires_cupy
def test_from_cupy(self, Var):
    if Var is IndexVariable:
        pytest.skip('cupy in default indexes is not supported at the moment')
    import cupy as cp
    arr = np.array([1, 2, 3])
    v = Var('x', cp.array(arr))
    assert_identical(v.as_numpy(), Var('x', arr))
    np.testing.assert_equal(v.to_numpy(), arr)