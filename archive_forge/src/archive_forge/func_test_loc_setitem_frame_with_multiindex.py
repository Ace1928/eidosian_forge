import numpy as np
import pytest
from pandas.errors import (
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_loc_setitem_frame_with_multiindex(self, multiindex_dataframe_random_data):
    frame = multiindex_dataframe_random_data
    frame.loc[('bar', 'two'), 'B'] = 5
    assert frame.loc[('bar', 'two'), 'B'] == 5
    df = frame.copy()
    df.columns = list(range(3))
    df.loc[('bar', 'two'), 1] = 7
    assert df.loc[('bar', 'two'), 1] == 7