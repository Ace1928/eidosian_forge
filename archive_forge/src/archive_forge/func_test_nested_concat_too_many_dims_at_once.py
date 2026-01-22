from __future__ import annotations
from datetime import datetime
from itertools import product
import numpy as np
import pytest
from xarray import (
from xarray.core import dtypes
from xarray.core.combine import (
from xarray.tests import assert_equal, assert_identical, requires_cftime
from xarray.tests.test_dataset import create_test_data
@pytest.mark.xfail
def test_nested_concat_too_many_dims_at_once(self):
    objs = [Dataset({'x': [0], 'y': [1]}), Dataset({'y': [0], 'x': [1]})]
    with pytest.raises(ValueError, match='not equal across datasets'):
        combine_nested(objs, concat_dim='x', coords='minimal')