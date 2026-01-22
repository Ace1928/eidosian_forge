import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
import pandas.core.common as com
def test_sample_wont_accept_n_and_frac(self, obj):
    msg = 'Please enter a value for `frac` OR `n`, not both'
    with pytest.raises(ValueError, match=msg):
        obj.sample(n=3, frac=0.3)