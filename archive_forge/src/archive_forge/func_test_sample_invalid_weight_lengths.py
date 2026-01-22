import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
import pandas.core.common as com
def test_sample_invalid_weight_lengths(self, obj):
    msg = 'Weights and axis to be sampled must be of same length'
    with pytest.raises(ValueError, match=msg):
        obj.sample(n=3, weights=[0, 1])
    with pytest.raises(ValueError, match=msg):
        bad_weights = [0.5] * 11
        obj.sample(n=3, weights=bad_weights)
    with pytest.raises(ValueError, match='Fewer non-zero entries in p than size'):
        bad_weight_series = Series([0, 0, 0.2])
        obj.sample(n=4, weights=bad_weight_series)