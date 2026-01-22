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
@pytest.mark.parametrize('data', test_string_list_data_values, ids=test_string_list_data_keys)
@pytest.mark.parametrize('sep', string_sep_values, ids=string_sep_keys)
def test_str_get_dummies(data, sep):
    modin_series, pandas_series = create_test_series(data)
    if sep:
        with warns_that_defaulting_to_pandas():
            modin_series.str.get_dummies(sep)