import datetime
from datetime import timedelta
import re
import numpy as np
import pytest
from pandas._libs.tslibs import Timestamp
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
from pandas.tests.io.pytables.common import (
def test_append_misc_empty_frame(setup_path):
    with ensure_clean_store(setup_path) as store:
        df_empty = DataFrame(columns=list('ABC'))
        store.append('df', df_empty)
        with pytest.raises(KeyError, match="'No object named df in the file'"):
            store.select('df')
        df = DataFrame(np.random.default_rng(2).random((10, 3)), columns=list('ABC'))
        store.append('df', df)
        tm.assert_frame_equal(store.select('df'), df)
        store.append('df', df_empty)
        tm.assert_frame_equal(store.select('df'), df)
        df = DataFrame(columns=list('ABC'))
        store.put('df2', df)
        tm.assert_frame_equal(store.select('df2'), df)