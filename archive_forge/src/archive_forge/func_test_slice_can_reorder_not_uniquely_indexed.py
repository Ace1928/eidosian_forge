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
def test_slice_can_reorder_not_uniquely_indexed():
    ser = Series(1, index=['a', 'a', 'b', 'b', 'c'])
    ser[::-1]