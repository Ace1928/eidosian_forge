from datetime import (
import numpy as np
import pytest
from pandas._libs.tslibs import (
from pandas.core.dtypes.common import is_scalar
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.indexing import IndexingError
from pandas.tseries.offsets import BDay
def test_getitem_boolean_contiguous_preserve_freq(self):
    rng = date_range('1/1/2000', '3/1/2000', freq='B')
    mask = np.zeros(len(rng), dtype=bool)
    mask[10:20] = True
    masked = rng[mask]
    expected = rng[10:20]
    assert expected.freq == rng.freq
    tm.assert_index_equal(masked, expected)
    mask[22] = True
    masked = rng[mask]
    assert masked.freq is None