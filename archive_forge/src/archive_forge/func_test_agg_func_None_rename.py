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
@pytest.mark.parametrize('by_and_agg_dict', [{'by': [list(test_data['int_data'].keys())[0], list(test_data['int_data'].keys())[1]], 'agg_dict': {'max': (list(test_data['int_data'].keys())[2], np.max), 'min': (list(test_data['int_data'].keys())[2], np.min)}}, {'by': ['col1'], 'agg_dict': {'max': (list(test_data['int_data'].keys())[0], np.max), 'min': (list(test_data['int_data'].keys())[-1], np.min)}}, {'by': [list(test_data['int_data'].keys())[0], list(test_data['int_data'].keys())[-1]], 'agg_dict': {'max': (list(test_data['int_data'].keys())[1], max), 'min': (list(test_data['int_data'].keys())[-2], min)}}, pytest.param({'by': [list(test_data['int_data'].keys())[0], list(test_data['int_data'].keys())[-1]], 'agg_dict': {'max': (list(test_data['int_data'].keys())[1], max), 'min': (list(test_data['int_data'].keys())[-1], min)}}, marks=pytest.mark.skip('See Modin issue #3602'))])
@pytest.mark.parametrize('as_index', [True, False])
def test_agg_func_None_rename(by_and_agg_dict, as_index):
    modin_df, pandas_df = create_test_dfs(test_data['int_data'])
    modin_result = modin_df.groupby(by_and_agg_dict['by'], as_index=as_index).agg(**by_and_agg_dict['agg_dict'])
    pandas_result = pandas_df.groupby(by_and_agg_dict['by'], as_index=as_index).agg(**by_and_agg_dict['agg_dict'])
    df_equals(modin_result, pandas_result)