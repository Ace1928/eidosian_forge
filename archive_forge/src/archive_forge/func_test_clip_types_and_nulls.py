from datetime import datetime
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_clip_types_and_nulls(self):
    sers = [Series([np.nan, 1.0, 2.0, 3.0]), Series([None, 'a', 'b', 'c']), Series(pd.to_datetime([np.nan, 1, 2, 3], unit='D'))]
    for s in sers:
        thresh = s[2]
        lower = s.clip(lower=thresh)
        upper = s.clip(upper=thresh)
        assert lower[notna(lower)].min() == thresh
        assert upper[notna(upper)].max() == thresh
        assert list(isna(s)) == list(isna(lower))
        assert list(isna(s)) == list(isna(upper))