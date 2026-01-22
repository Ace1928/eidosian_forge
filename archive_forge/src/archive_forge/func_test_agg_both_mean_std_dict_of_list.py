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
def test_agg_both_mean_std_dict_of_list(cases, a_mean, a_std):
    expected = pd.concat([a_mean, a_std], axis=1)
    expected.columns = pd.MultiIndex.from_tuples([('A', 'mean'), ('A', 'std')])
    result = cases.aggregate({'A': ['mean', 'std']})
    tm.assert_frame_equal(result, expected)