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
def inter_df_math_helper(modin_series, pandas_series, op, comparator_kwargs=None, expected_exception=None):
    inter_df_math_helper_one_side(modin_series, pandas_series, op, comparator_kwargs, expected_exception)
    rop = get_rop(op)
    if rop:
        inter_df_math_helper_one_side(modin_series, pandas_series, rop, comparator_kwargs, expected_exception)