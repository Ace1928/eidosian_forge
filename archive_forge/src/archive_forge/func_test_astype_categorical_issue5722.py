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
@pytest.mark.parametrize('data', [['a', 'a', 'b', 'c', 'c', 'd', 'b', 'd']])
@pytest.mark.parametrize('set_min_partition_size', [2, 4], ids=['four_partitions', 'two_partitions'], indirect=True)
def test_astype_categorical_issue5722(data, set_min_partition_size):
    modin_series, pandas_series = create_test_series(data)
    modin_result = modin_series.astype('category')
    pandas_result = pandas_series.astype('category')
    df_equals(modin_result, pandas_result)
    assert modin_result.dtype == pandas_result.dtype
    pandas_result1, pandas_result2 = (pandas_result.iloc[:4], pandas_result.iloc[4:])
    modin_result1, modin_result2 = (modin_result.iloc[:4], modin_result.iloc[4:])
    assert pandas_result1.cat.categories.equals(pandas_result2.cat.categories)
    assert modin_result1.cat.categories.equals(modin_result2.cat.categories)
    assert pandas_result1.cat.categories.equals(modin_result1.cat.categories)
    assert pandas_result2.cat.categories.equals(modin_result2.cat.categories)
    assert_array_equal(pandas_result1.cat.codes.values, modin_result1.cat.codes.values)
    assert_array_equal(pandas_result2.cat.codes.values, modin_result2.cat.codes.values)