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
def test___setattr__not_column():
    pandas_df = pandas.DataFrame([1, 2, 3])
    modin_df = pd.DataFrame([1, 2, 3])
    pandas_df.new_col = [4, 5, 6]
    modin_df.new_col = [4, 5, 6]
    df_equals(modin_df, pandas_df)
    assert modin_df.new_col == pandas_df.new_col