from __future__ import annotations
import datetime
import itertools
import json
import unittest.mock as mock
import matplotlib
import numpy as np
import pandas
import pandas._libs.lib as lib
import pytest
from numpy.testing import assert_array_equal
from pandas.core.indexing import IndexingError
from pandas.errors import SpecificationError
import modin.pandas as pd
from modin.config import Engine, NPartitions, StorageFormat
from modin.pandas.io import to_pandas
from modin.pandas.testing import assert_series_equal
from modin.tests.test_utils import warns_that_defaulting_to_pandas
from modin.utils import get_current_execution, try_cast_to_pandas
from .utils import (
@pytest.mark.parametrize('key', [pytest.param(lambda idx: slice(1, 3), id='location_based_slice'), pytest.param(lambda idx: slice(idx[1], idx[-1]), id='index_based_slice'), pytest.param(lambda idx: [idx[0], idx[2], idx[-1]], id='list_of_labels'), pytest.param(lambda idx: [True if i % 2 else False for i in range(len(idx))], id='boolean_mask')])
@pytest.mark.parametrize('index', [pytest.param(lambda idx_len: [chr(x) for x in range(ord('a'), ord('a') + idx_len)], id='str_index'), pytest.param(lambda idx_len: list(range(1, idx_len + 1)), id='int_index')])
def test___setitem___non_hashable(key, index):
    data = np.arange(5)
    index = index(len(data))
    key = key(index)
    md_sr, pd_sr = create_test_series(data, index=index)
    md_sr[key] = 10
    pd_sr[key] = 10
    df_equals(md_sr, pd_sr)