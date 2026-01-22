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
def eval_shift(modin_groupby, pandas_groupby, comparator=None):
    if comparator is None:

        def comparator(df1, df2):
            df_equals(*sort_if_experimental_groupby(df1, df2))
    eval_general(modin_groupby, pandas_groupby, lambda groupby: groupby.shift(), comparator=comparator)
    eval_general(modin_groupby, pandas_groupby, lambda groupby: groupby.shift(periods=0), comparator=comparator)
    eval_general(modin_groupby, pandas_groupby, lambda groupby: groupby.shift(periods=-3), comparator=comparator)
    if get_current_execution() != 'BaseOnPython':
        if isinstance(pandas_groupby, pandas.core.groupby.DataFrameGroupBy):
            pandas_res = pandas_groupby.shift(axis=1, fill_value=777)
            modin_res = modin_groupby.shift(axis=1, fill_value=777)
            import pandas.core.algorithms as algorithms
            indexer, _ = modin_res.index.get_indexer_non_unique(modin_res.index._values)
            indexer = algorithms.unique1d(indexer)
            modin_res = modin_res.take(indexer)
            comparator(modin_res, pandas_res)
        else:
            eval_general(modin_groupby, pandas_groupby, lambda groupby: groupby.shift(fill_value=777), comparator=comparator)