import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
import pandas.core.common as com
def test_sample_requires_positive_n_frac(self, obj):
    with pytest.raises(ValueError, match='A negative number of rows requested. Please provide `n` >= 0'):
        obj.sample(n=-3)
    with pytest.raises(ValueError, match='A negative number of rows requested. Please provide `frac` >= 0'):
        obj.sample(frac=-0.3)