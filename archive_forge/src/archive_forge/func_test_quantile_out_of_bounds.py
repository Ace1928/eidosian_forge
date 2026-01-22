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
@pytest.mark.parametrize('compute_backend', ['numbagg', None], indirect=True)
@pytest.mark.parametrize('q', [-0.1, 1.1, [2], [0.25, 2]])
def test_quantile_out_of_bounds(self, q, compute_backend):
    v = Variable(['x', 'y'], self.d)
    with pytest.raises(ValueError, match='(Q|q)uantiles must be in the range \\[0, 1\\]'):
        v.quantile(q, dim='x')