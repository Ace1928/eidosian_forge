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
@pytest.mark.skipif(get_current_execution() == 'BaseOnPython' or StorageFormat.get() == 'Hdk', reason='The test only make sense for partitioned executions')
def test_groupby_with_virtual_partitions():
    modin_df, pandas_df = create_test_dfs(test_data['int_data'])
    big_modin_df = pd.concat([modin_df for _ in range(5)])
    big_pandas_df = pandas.concat([pandas_df for _ in range(5)])
    assert issubclass(type(big_modin_df._query_compiler._modin_frame._partitions[0][0]), PandasDataframeAxisPartition)
    eval_general(big_modin_df, big_pandas_df, lambda df: df.groupby(df.columns[0]).count())