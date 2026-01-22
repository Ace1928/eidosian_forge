import numpy as np
import pytest
from pandas.core.dtypes.cast import find_common_type
from pandas.core.dtypes.common import pandas_dtype
from pandas.core.dtypes.dtypes import (
from pandas import (
def test_raises_empty_input():
    with pytest.raises(ValueError, match='no types given'):
        find_common_type([])