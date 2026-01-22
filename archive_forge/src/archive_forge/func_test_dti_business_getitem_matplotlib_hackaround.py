from datetime import (
import numpy as np
import pytest
from pandas._libs import index as libindex
from pandas.compat.numpy import np_long
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tseries.frequencies import to_offset
@pytest.mark.parametrize('freq', ['B', 'C'])
def test_dti_business_getitem_matplotlib_hackaround(self, freq):
    rng = bdate_range(START, END, freq=freq)
    with pytest.raises(ValueError, match='Multi-dimensional indexing'):
        rng[:, None]