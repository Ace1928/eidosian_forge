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
def test_unique_pyarrow_dtype():
    modin_series, pandas_series = create_test_series([1, 0, pd.NA], dtype='uint8[pyarrow]')

    def comparator(df1, df2):
        df_equals(df1, df2)
        assert type(df1) is type(df2)
    eval_general(modin_series, pandas_series, lambda df: df.unique(), comparator=comparator)