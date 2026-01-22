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
def test_indexer_type(self):
    data = np.random.random((10, 11))
    v = Variable(['x', 'y'], data)

    def assert_indexer_type(key, object_type):
        dims, index_tuple, new_order = v._broadcast_indexes(key)
        assert isinstance(index_tuple, object_type)
    assert_indexer_type((0, 1), BasicIndexer)
    assert_indexer_type((0, slice(None, None)), BasicIndexer)
    assert_indexer_type((Variable([], 3), slice(None, None)), BasicIndexer)
    assert_indexer_type((Variable([], 3), Variable([], 6)), BasicIndexer)
    assert_indexer_type(([0, 1], 1), OuterIndexer)
    assert_indexer_type(([0, 1], [1, 2]), OuterIndexer)
    assert_indexer_type((Variable('x', [0, 1]), 1), OuterIndexer)
    assert_indexer_type((Variable('x', [0, 1]), slice(None, None)), OuterIndexer)
    assert_indexer_type((Variable('x', [0, 1]), Variable('y', [0, 1])), OuterIndexer)
    assert_indexer_type((Variable('y', [0, 1]), [0, 1]), VectorizedIndexer)
    assert_indexer_type((Variable('z', [0, 1]), Variable('z', [0, 1])), VectorizedIndexer)
    assert_indexer_type((Variable(('a', 'b'), [[0, 1], [1, 2]]), Variable(('a', 'b'), [[0, 1], [1, 2]])), VectorizedIndexer)