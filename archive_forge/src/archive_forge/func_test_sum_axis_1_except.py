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
@pytest.mark.parametrize('operation', ['sum', 'shift'])
def test_sum_axis_1_except(operation):
    eval_general(*create_test_series(test_data['int_data']), lambda df, *args, **kwargs: getattr(df, operation)(*args, **kwargs), axis=1, expected_exception=ValueError('No axis named 1 for object type Series'))