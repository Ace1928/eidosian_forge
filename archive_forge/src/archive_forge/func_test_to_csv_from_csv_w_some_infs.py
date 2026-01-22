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
def test_to_csv_from_csv_w_some_infs(self, float_frame):
    float_frame['G'] = np.nan
    f = lambda x: [np.inf, np.nan][np.random.default_rng(2).random() < 0.5]
    float_frame['h'] = float_frame.index.map(f)
    with tm.ensure_clean() as path:
        float_frame.to_csv(path)
        recons = self.read_csv(path)
        tm.assert_frame_equal(float_frame, recons)
        tm.assert_frame_equal(np.isinf(float_frame), np.isinf(recons))