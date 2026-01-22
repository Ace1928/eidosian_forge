from string import ascii_letters
import numpy as np
import pytest
from pandas.errors import (
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.arm_slow
def test_detect_chained_assignment_changing_dtype(self, using_array_manager, using_copy_on_write, warn_copy_on_write):
    df = DataFrame({'A': date_range('20130101', periods=5), 'B': np.random.default_rng(2).standard_normal(5), 'C': np.arange(5, dtype='int64'), 'D': ['a', 'b', 'c', 'd', 'e']})
    df_original = df.copy()
    if using_copy_on_write or warn_copy_on_write:
        with tm.raises_chained_assignment_error():
            df.loc[2]['D'] = 'foo'
        with tm.raises_chained_assignment_error():
            df.loc[2]['C'] = 'foo'
        tm.assert_frame_equal(df, df_original)
        with tm.raises_chained_assignment_error(extra_warnings=(FutureWarning,)):
            df['C'][2] = 'foo'
        if using_copy_on_write:
            tm.assert_frame_equal(df, df_original)
        else:
            assert df.loc[2, 'C'] == 'foo'
    else:
        with pytest.raises(SettingWithCopyError, match=msg):
            df.loc[2]['D'] = 'foo'
        with pytest.raises(SettingWithCopyError, match=msg):
            df.loc[2]['C'] = 'foo'
        if not using_array_manager:
            with pytest.raises(SettingWithCopyError, match=msg):
                with tm.raises_chained_assignment_error():
                    df['C'][2] = 'foo'
        else:
            df['C'][2] = 'foo'
            assert df.loc[2, 'C'] == 'foo'