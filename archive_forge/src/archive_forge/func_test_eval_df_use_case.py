import matplotlib
import numpy as np
import pandas
import pytest
from pandas.core.dtypes.common import is_list_like
import modin.pandas as pd
from modin.config import MinPartitionSize, NPartitions
from modin.tests.pandas.utils import (
from modin.tests.test_utils import warns_that_defaulting_to_pandas
from modin.utils import get_current_execution
def test_eval_df_use_case():
    frame_data = {'a': random_state.randn(10), 'b': random_state.randn(10)}
    df = pandas.DataFrame(frame_data)
    modin_df = pd.DataFrame(frame_data)
    tmp_pandas = df.eval('arctan2(sin(a), b)', engine='python', parser='pandas')
    tmp_modin = modin_df.eval('arctan2(sin(a), b)', engine='python', parser='pandas')
    assert isinstance(tmp_modin, pd.Series)
    df_equals(tmp_modin, tmp_pandas)
    tmp_pandas = df.eval('e = arctan2(sin(a), b)', engine='python', parser='pandas')
    tmp_modin = modin_df.eval('e = arctan2(sin(a), b)', engine='python', parser='pandas')
    df_equals(tmp_modin, tmp_pandas)
    df.eval('e = arctan2(sin(a), b)', engine='python', parser='pandas', inplace=True)
    modin_df.eval('e = arctan2(sin(a), b)', engine='python', parser='pandas', inplace=True)
    df_equals(modin_df, df)