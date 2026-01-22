from __future__ import annotations
import collections
from operator import add
import numpy as np
import pytest
import dask
import dask.array as da
from dask.array.utils import assert_eq
from dask.blockwise import (
from dask.highlevelgraph import HighLevelGraph
from dask.utils_test import dec, hlg_layer_topological, inc
@pytest.mark.parametrize('name', ['_0', '_1', '.', '.0', '_'])
def test_common_token_names_kwargs(name):
    x = np.array(['a', 'bb', 'ccc'], dtype=object)
    d = da.from_array(x, chunks=2)
    result = da.blockwise(lambda x, y: x + y, 'i', d, 'i', y=name, dtype=object)
    expected = x + name
    assert_eq(result, expected)