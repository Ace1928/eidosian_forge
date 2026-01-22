import io
import warnings
import matplotlib
import numpy as np
import pandas
import pandas._libs.lib as lib
import pytest
from numpy.testing import assert_array_equal
import modin.pandas as pd
from modin.config import Engine, NPartitions, StorageFormat
from modin.pandas.io import to_pandas
from modin.tests.pandas.utils import (
from modin.tests.test_utils import warns_that_defaulting_to_pandas
from modin.utils import get_current_execution
def test_between_time():
    i = pd.date_range('2008-01-01', periods=1000, freq='12H')
    modin_df = pd.DataFrame({'A': list(range(1000)), 'B': list(range(1000))}, index=i)
    pandas_df = pandas.DataFrame({'A': list(range(1000)), 'B': list(range(1000))}, index=i)
    df_equals(modin_df.between_time('12:00', '17:00'), pandas_df.between_time('12:00', '17:00'))
    df_equals(modin_df.between_time('3:00', '4:00'), pandas_df.between_time('3:00', '4:00'))
    df_equals(modin_df.T.between_time('12:00', '17:00', axis=1), pandas_df.T.between_time('12:00', '17:00', axis=1))