import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_case_when_odd_caselist(df):
    """
    Raise ValueError if no of caselist is odd.
    """
    msg = 'Argument 0 must have length 2; '
    msg += 'a condition and replacement; instead got length 3.'
    with pytest.raises(ValueError, match=msg):
        df['a'].case_when([(df['a'].eq(1), 1, df.a.gt(1))])