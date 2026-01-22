import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_case_when_raise_error_from_mask(df):
    """
    Raise Error from within Series.mask
    """
    msg = 'Failed to apply condition0 and replacement0.'
    with pytest.raises(ValueError, match=msg):
        df['a'].case_when([(df['a'].eq(1), [1, 2])])