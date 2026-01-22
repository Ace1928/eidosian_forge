import datetime
import itertools
from unittest import mock
import numpy as np
import pandas
import pandas._libs.lib as lib
import pytest
import modin.pandas as pd
from modin.config import (
from modin.core.dataframe.algebra.default2pandas.groupby import GroupBy
from modin.core.dataframe.pandas.partitioning.axis_partition import (
from modin.pandas.io import from_pandas
from modin.pandas.utils import is_scalar
from modin.tests.test_utils import warns_that_defaulting_to_pandas
from modin.utils import (
from .utils import (
def test_skew_corner_cases():
    """
    This test was inspired by https://github.com/modin-project/modin/issues/5545.

    The test verifies that modin acts exactly as pandas when the input data is
    bad for the 'skew' and so some components of the 'skew' formula appears to be invalid:
        ``(count * (count - 1) ** 0.5 / (count - 2)) * (m3 / m2**1.5)``
    """
    modin_df, pandas_df = create_test_dfs({'col0': [1, 1, 1], 'col1': [10, 10, 10]})
    eval_general(modin_df, pandas_df, lambda df: df.groupby('col0').skew())
    modin_df, pandas_df = create_test_dfs({'col0': [1, 1], 'col1': [1, 2]})
    eval_general(modin_df, pandas_df, lambda df: df.groupby('col0').skew())
    modin_df, pandas_df = create_test_dfs({'col0': [1, 1], 'col1': [171, 137]})
    eval_general(modin_df, pandas_df, lambda df: df.groupby('col0').skew())