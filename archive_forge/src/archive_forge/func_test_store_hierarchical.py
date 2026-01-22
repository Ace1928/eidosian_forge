import datetime
import re
import numpy as np
import pytest
from pandas._libs.tslibs import Timestamp
from pandas.compat import is_platform_windows
import pandas as pd
from pandas import (
from pandas.tests.io.pytables.common import (
from pandas.util import _test_decorators as td
def test_store_hierarchical(setup_path, multiindex_dataframe_random_data):
    frame = multiindex_dataframe_random_data
    _check_roundtrip(frame, tm.assert_frame_equal, path=setup_path)
    _check_roundtrip(frame.T, tm.assert_frame_equal, path=setup_path)
    _check_roundtrip(frame['A'], tm.assert_series_equal, path=setup_path)
    with ensure_clean_store(setup_path) as store:
        store['frame'] = frame
        recons = store['frame']
        tm.assert_frame_equal(recons, frame)