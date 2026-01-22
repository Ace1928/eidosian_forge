import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
def assert_invalid_addsub_type(left, right, msg=None):
    """
    Helper to assert that left and right can be neither added nor subtracted.

    Parameters
    ----------
    left : object
    right : object
    msg : str or None, default None
    """
    with pytest.raises(TypeError, match=msg):
        left + right
    with pytest.raises(TypeError, match=msg):
        right + left
    with pytest.raises(TypeError, match=msg):
        left - right
    with pytest.raises(TypeError, match=msg):
        right - left