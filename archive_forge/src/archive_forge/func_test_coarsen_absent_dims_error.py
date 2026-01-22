from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray import DataArray, Dataset, set_options
from xarray.core import duck_array_ops
from xarray.tests import (
def test_coarsen_absent_dims_error(ds: Dataset) -> None:
    with pytest.raises(ValueError, match="Window dimensions \\('foo',\\) not found in Dataset dimensions"):
        ds.coarsen(foo=2)