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
@pytest.mark.parametrize('data', [['A', 'A', 'B', 'B', 'A'], [1, 1, 2, 1, 2, 2, 3, 1, 2, 1, 2]])
def test_astype_categorical(data):
    modin_df, pandas_df = create_test_series(data)
    modin_result = modin_df.astype('category')
    pandas_result = pandas_df.astype('category')
    df_equals(modin_result, pandas_result)
    assert modin_result.dtype == pandas_result.dtype