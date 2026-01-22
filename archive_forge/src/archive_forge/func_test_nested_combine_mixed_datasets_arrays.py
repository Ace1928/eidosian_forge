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
def test_nested_combine_mixed_datasets_arrays(self):
    objs = [DataArray([0, 1], dims='x', coords={'x': [0, 1]}), Dataset({'x': [2, 3]})]
    with pytest.raises(ValueError, match="Can't combine datasets with unnamed arrays."):
        combine_nested(objs, 'x')