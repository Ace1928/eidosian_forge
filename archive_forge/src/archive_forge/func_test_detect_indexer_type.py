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
def test_detect_indexer_type(self):
    """Tests indexer type was correctly detected."""
    data = np.random.random((10, 11))
    v = Variable(['x', 'y'], data)
    _, ind, _ = v._broadcast_indexes((0, 1))
    assert type(ind) == indexing.BasicIndexer
    _, ind, _ = v._broadcast_indexes((0, slice(0, 8, 2)))
    assert type(ind) == indexing.BasicIndexer
    _, ind, _ = v._broadcast_indexes((0, [0, 1]))
    assert type(ind) == indexing.OuterIndexer
    _, ind, _ = v._broadcast_indexes(([0, 1], 1))
    assert type(ind) == indexing.OuterIndexer
    _, ind, _ = v._broadcast_indexes(([0, 1], [1, 2]))
    assert type(ind) == indexing.OuterIndexer
    _, ind, _ = v._broadcast_indexes(([0, 1], slice(0, 8, 2)))
    assert type(ind) == indexing.OuterIndexer
    vind = Variable(('a',), [0, 1])
    _, ind, _ = v._broadcast_indexes((vind, slice(0, 8, 2)))
    assert type(ind) == indexing.OuterIndexer
    vind = Variable(('y',), [0, 1])
    _, ind, _ = v._broadcast_indexes((vind, 3))
    assert type(ind) == indexing.OuterIndexer
    vind = Variable(('a',), [0, 1])
    _, ind, _ = v._broadcast_indexes((vind, vind))
    assert type(ind) == indexing.VectorizedIndexer
    vind = Variable(('a', 'b'), [[0, 2], [1, 3]])
    _, ind, _ = v._broadcast_indexes((vind, 3))
    assert type(ind) == indexing.VectorizedIndexer