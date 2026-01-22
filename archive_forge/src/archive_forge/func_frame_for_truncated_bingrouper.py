from datetime import (
import numpy as np
import pytest
import pytz
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.groupby.grouper import Grouper
from pandas.core.groupby.ops import BinGrouper
@pytest.fixture
def frame_for_truncated_bingrouper():
    """
    DataFrame used by groupby_with_truncated_bingrouper, made into
    a separate fixture for easier reuse in
    test_groupby_apply_timegrouper_with_nat_apply_squeeze
    """
    df = DataFrame({'Quantity': [18, 3, 5, 1, 9, 3], 'Date': [Timestamp(2013, 9, 1, 13, 0), Timestamp(2013, 9, 1, 13, 5), Timestamp(2013, 10, 1, 20, 0), Timestamp(2013, 10, 3, 10, 0), pd.NaT, Timestamp(2013, 9, 2, 14, 0)]})
    return df