import io
import warnings
import matplotlib
import numpy as np
import pandas
import pytest
import modin.pandas as pd
from modin.config import NPartitions
from modin.pandas.utils import SET_DATAFRAME_ATTRIBUTE_WARNING
from modin.tests.pandas.utils import (
from modin.tests.test_utils import warns_that_defaulting_to_pandas
def test_itertuples_multiindex():
    data = test_data['int_data']
    modin_df, pandas_df = (pd.DataFrame(data), pandas.DataFrame(data))
    new_idx = pd.MultiIndex.from_tuples([(i // 4, i // 2, i) for i in range(len(modin_df.columns))])
    modin_df.columns = new_idx
    pandas_df.columns = new_idx
    modin_it_custom = modin_df.itertuples()
    pandas_it_custom = pandas_df.itertuples()
    for modin_row, pandas_row in zip(modin_it_custom, pandas_it_custom):
        np.testing.assert_equal(modin_row, pandas_row)