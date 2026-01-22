import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
import pandas.core.common as com
def test_sample_zero_weights(self, obj):
    zero_weights = [0] * 10
    with pytest.raises(ValueError, match='Invalid weights: weights sum to zero'):
        obj.sample(n=3, weights=zero_weights)