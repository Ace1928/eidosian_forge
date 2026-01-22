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
def test_groupby_on_index_values_with_loop():
    length = 2 ** 6
    data = {'a': np.random.randint(0, 100, size=length), 'b': np.random.randint(0, 100, size=length), 'c': np.random.randint(0, 100, size=length)}
    idx = ['g1' if i % 3 != 0 else 'g2' for i in range(length)]
    modin_df = pd.DataFrame(data, index=idx, columns=list('aba'))
    pandas_df = pandas.DataFrame(data, index=idx, columns=list('aba'))
    modin_groupby_obj = modin_df.groupby(modin_df.index)
    pandas_groupby_obj = pandas_df.groupby(pandas_df.index)
    modin_dict = {k: v for k, v in modin_groupby_obj}
    pandas_dict = {k: v for k, v in pandas_groupby_obj}
    for k in modin_dict:
        df_equals(modin_dict[k], pandas_dict[k])
    modin_groupby_obj = modin_df.groupby(modin_df.columns, axis=1)
    pandas_groupby_obj = pandas_df.groupby(pandas_df.columns, axis=1)
    modin_dict = {k: v for k, v in modin_groupby_obj}
    pandas_dict = {k: v for k, v in pandas_groupby_obj}
    for k in modin_dict:
        df_equals(modin_dict[k], pandas_dict[k])