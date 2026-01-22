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
def test_isin_with_series():
    modin_series1, pandas_series1 = create_test_series([1, 2, 3])
    modin_series2, pandas_series2 = create_test_series([1, 2, 3, 4, 5])
    eval_general((modin_series1, modin_series2), (pandas_series1, pandas_series2), lambda srs: srs[0].isin(srs[1]))
    modin_series1, pandas_series1 = create_test_series([1, 2, 3], index=[10, 11, 12])
    eval_general((modin_series1, modin_series2), (pandas_series1, pandas_series2), lambda srs: srs[0].isin(srs[1]))