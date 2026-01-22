from __future__ import annotations
import sys
from textwrap import dedent
import numpy as np
import pandas as pd
import pytest
from numpy.core import defchararray
import xarray as xr
from xarray.core import formatting
from xarray.tests import requires_cftime, requires_dask, requires_netCDF4
def test_get_indexer_at_least_n_items(self) -> None:
    cases = [((20,), (slice(10),), (slice(-10, None),)), ((3, 20), (0, slice(10)), (-1, slice(-10, None))), ((2, 10), (0, slice(10)), (-1, slice(-10, None))), ((2, 5), (slice(2), slice(None)), (slice(-2, None), slice(None))), ((1, 2, 5), (0, slice(2), slice(None)), (-1, slice(-2, None), slice(None))), ((2, 3, 5), (0, slice(2), slice(None)), (-1, slice(-2, None), slice(None))), ((1, 10, 1), (0, slice(10), slice(None)), (-1, slice(-10, None), slice(None))), ((2, 5, 1), (slice(2), slice(None), slice(None)), (slice(-2, None), slice(None), slice(None))), ((2, 5, 3), (0, slice(4), slice(None)), (-1, slice(-4, None), slice(None))), ((2, 3, 3), (slice(2), slice(None), slice(None)), (slice(-2, None), slice(None), slice(None)))]
    for shape, start_expected, end_expected in cases:
        actual = formatting._get_indexer_at_least_n_items(shape, 10, from_end=False)
        assert start_expected == actual
        actual = formatting._get_indexer_at_least_n_items(shape, 10, from_end=True)
        assert end_expected == actual