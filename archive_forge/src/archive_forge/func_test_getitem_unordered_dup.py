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
def test_getitem_unordered_dup():
    obj = Series(range(5), index=['c', 'a', 'a', 'b', 'b'])
    assert is_scalar(obj['c'])
    assert obj['c'] == 0