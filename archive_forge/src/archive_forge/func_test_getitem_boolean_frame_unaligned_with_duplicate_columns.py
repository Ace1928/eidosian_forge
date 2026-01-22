import re
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import SparseArray
def test_getitem_boolean_frame_unaligned_with_duplicate_columns(self, df_dup_cols):
    df = df_dup_cols
    msg = 'cannot reindex on an axis with duplicate labels'
    with pytest.raises(ValueError, match=msg):
        df[df.A > 6]