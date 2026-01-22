from datetime import datetime
import itertools
import re
import numpy as np
import pytest
from pandas._libs import lib
from pandas.errors import PerformanceWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.reshape import reshape as reshape_lib
def test_unstack_level_name(self, multiindex_dataframe_random_data):
    frame = multiindex_dataframe_random_data
    result = frame.unstack('second')
    expected = frame.unstack(level=1)
    tm.assert_frame_equal(result, expected)