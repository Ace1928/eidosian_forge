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
@pytest.mark.parametrize('data', test_string_data_values, ids=test_string_data_keys)
@pytest.mark.parametrize('pat', string_sep_values, ids=string_sep_keys)
@pytest.mark.parametrize('na', string_na_rep_values, ids=string_na_rep_keys)
def test_str_endswith(data, pat, na):
    modin_series, pandas_series = create_test_series(data)
    eval_general(modin_series, pandas_series, lambda series: series.str.endswith(pat, na=na), comparator_kwargs={'check_dtypes': False})