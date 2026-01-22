from datetime import datetime
from itertools import product
import numpy as np
import pytest
from pandas.core.dtypes.common import (
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_errorreset_index_rename(float_frame):
    stacked_df = float_frame.stack(future_stack=True)[::2]
    stacked_df = DataFrame({'first': stacked_df, 'second': stacked_df})
    with pytest.raises(ValueError, match='Index names must be str or 1-dimensional list'):
        stacked_df.reset_index(names={'first': 'new_first', 'second': 'new_second'})
    with pytest.raises(IndexError, match='list index out of range'):
        stacked_df.reset_index(names=['new_first'])