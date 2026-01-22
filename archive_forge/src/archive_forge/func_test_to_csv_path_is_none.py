import csv
from io import StringIO
import os
import numpy as np
import pytest
from pandas.errors import ParserError
import pandas as pd
from pandas import (
import pandas._testing as tm
import pandas.core.common as com
from pandas.io.common import get_handle
def test_to_csv_path_is_none(self, float_frame):
    csv_str = float_frame.to_csv(path_or_buf=None)
    assert isinstance(csv_str, str)
    recons = read_csv(StringIO(csv_str), index_col=0)
    tm.assert_frame_equal(float_frame, recons)