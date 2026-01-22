from datetime import (
import numpy as np
import pytest
import pytz
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.groupby.grouper import Grouper
from pandas.core.groupby.ops import BinGrouper
def test_frame_datetime64_handling_groupby(self):
    df = DataFrame([(3, np.datetime64('2012-07-03')), (3, np.datetime64('2012-07-04'))], columns=['a', 'date'])
    result = df.groupby('a').first()
    assert result['date'][3] == Timestamp('2012-07-03')