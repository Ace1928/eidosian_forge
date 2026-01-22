from __future__ import annotations
import operator
import pickle
import sys
from contextlib import suppress
from textwrap import dedent
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray import DataArray, Dataset, Variable
from xarray.core import duck_array_ops
from xarray.core.duck_array_ops import lazy_array_equiv
from xarray.testing import assert_chunks_equal
from xarray.tests import (
from xarray.tests.test_backends import create_tmp_file
@pytest.mark.parametrize('obj', [make_da()])
def test_auto_chunk_da(obj):
    actual = obj.chunk('auto').data
    expected = obj.data.rechunk('auto')
    np.testing.assert_array_equal(actual, expected)
    assert actual.chunks == expected.chunks