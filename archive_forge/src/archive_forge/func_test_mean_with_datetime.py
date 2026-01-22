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
@pytest.mark.parametrize('by_func', [lambda df: 'timestamp0', lambda df: ['timestamp0', 'timestamp1'], lambda df: ['timestamp0', df['timestamp1']]])
def test_mean_with_datetime(by_func):
    data = {'timestamp0': [pd.to_datetime(1490195805, unit='s')], 'timestamp1': [pd.to_datetime(1490195805, unit='s')], 'numeric': [0]}
    modin_df, pandas_df = create_test_dfs(data)
    eval_general(modin_df, pandas_df, lambda df: df.groupby(by=by_func(df)).mean())