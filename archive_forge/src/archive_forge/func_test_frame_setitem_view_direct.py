import numpy as np
import pytest
from pandas.errors import SettingWithCopyError
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
@td.skip_array_manager_invalid_test
def test_frame_setitem_view_direct(multiindex_dataframe_random_data, using_copy_on_write):
    df = multiindex_dataframe_random_data.T
    if using_copy_on_write:
        with pytest.raises(ValueError, match='read-only'):
            df['foo'].values[:] = 0
        assert (df['foo'].values != 0).all()
    else:
        df['foo'].values[:] = 0
        assert (df['foo'].values == 0).all()