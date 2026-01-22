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
def groupby_with_truncated_bingrouper(frame_for_truncated_bingrouper):
    """
    GroupBy object such that gb._grouper is a BinGrouper and
    len(gb._grouper.result_index) < len(gb._grouper.group_keys_seq)

    Aggregations on this groupby should have

        dti = date_range("2013-09-01", "2013-10-01", freq="5D", name="Date")

    As either the index or an index level.
    """
    df = frame_for_truncated_bingrouper
    tdg = Grouper(key='Date', freq='5D')
    gb = df.groupby(tdg)
    assert len(gb._grouper.result_index) != len(gb._grouper.group_keys_seq)
    return gb