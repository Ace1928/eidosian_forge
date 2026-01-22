from datetime import (
import re
import numpy as np
import pytest
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.reshape.concat import concat
from pandas.core.reshape.merge import (
def test_merge_indicator_invalid(self, dfs_for_indicator):
    df1, _ = dfs_for_indicator
    for i in ['_right_indicator', '_left_indicator', '_merge']:
        df_badcolumn = DataFrame({'col1': [1, 2], i: [2, 2]})
        msg = f'Cannot use `indicator=True` option when data contains a column named {i}|Cannot use name of an existing column for indicator column'
        with pytest.raises(ValueError, match=msg):
            merge(df1, df_badcolumn, on='col1', how='outer', indicator=True)
        with pytest.raises(ValueError, match=msg):
            df1.merge(df_badcolumn, on='col1', how='outer', indicator=True)
    df_badcolumn = DataFrame({'col1': [1, 2], 'custom_column_name': [2, 2]})
    msg = 'Cannot use name of an existing column for indicator column'
    with pytest.raises(ValueError, match=msg):
        merge(df1, df_badcolumn, on='col1', how='outer', indicator='custom_column_name')
    with pytest.raises(ValueError, match=msg):
        df1.merge(df_badcolumn, on='col1', how='outer', indicator='custom_column_name')