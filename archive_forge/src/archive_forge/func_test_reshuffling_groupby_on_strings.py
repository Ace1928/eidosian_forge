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
@pytest.mark.parametrize('modify_config', [{RangePartitioning: True}], indirect=True)
def test_reshuffling_groupby_on_strings(modify_config):
    modin_df, pandas_df = create_test_dfs({'col1': ['a'] * 50 + ['b'] * 50, 'col2': range(100)})
    modin_df = modin_df.astype({'col1': 'string'})
    pandas_df = pandas_df.astype({'col1': 'string'})
    md_grp = modin_df.groupby('col1')
    pd_grp = pandas_df.groupby('col1')
    eval_general(md_grp, pd_grp, lambda grp: grp.mean())
    eval_general(md_grp, pd_grp, lambda grp: grp.nth(2))
    eval_general(md_grp, pd_grp, lambda grp: grp.head(10))
    eval_general(md_grp, pd_grp, lambda grp: grp.tail(10))