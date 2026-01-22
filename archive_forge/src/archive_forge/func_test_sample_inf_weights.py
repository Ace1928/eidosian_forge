import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
import pandas.core.common as com
def test_sample_inf_weights(self, obj):
    weights_with_inf = [0.1] * 10
    weights_with_inf[0] = np.inf
    msg = 'weight vector may not include `inf` values'
    with pytest.raises(ValueError, match=msg):
        obj.sample(n=3, weights=weights_with_inf)
    weights_with_ninf = [0.1] * 10
    weights_with_ninf[0] = -np.inf
    with pytest.raises(ValueError, match=msg):
        obj.sample(n=3, weights=weights_with_ninf)