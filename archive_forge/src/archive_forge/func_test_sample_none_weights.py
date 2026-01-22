import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
import pandas.core.common as com
def test_sample_none_weights(self, obj):
    weights_with_None = [None] * 10
    weights_with_None[5] = 0.5
    tm.assert_equal(obj.sample(n=1, axis=0, weights=weights_with_None), obj.iloc[5:6])