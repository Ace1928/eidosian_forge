from datetime import datetime
import re
import numpy as np
import pytest
from pandas._libs import lib
from pandas.errors import UnsupportedFunctionCall
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.indexes.datetimes import date_range
@pytest.fixture
def df_mult(df_col, index):
    df_mult = df_col.copy()
    df_mult.index = pd.MultiIndex.from_arrays([range(10), index], names=['index', 'date'])
    return df_mult