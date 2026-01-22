from datetime import datetime
from itertools import product
import numpy as np
import pytest
from pandas.core.dtypes.common import (
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('flag', [False, True])
def test_reset_index_duplicate_columns_default(self, multiindex_df, flag):
    df = multiindex_df.rename_axis('A')
    df = df.set_flags(allows_duplicate_labels=flag)
    msg = "cannot insert \\('A', ''\\), already exists"
    with pytest.raises(ValueError, match=msg):
        df.reset_index()