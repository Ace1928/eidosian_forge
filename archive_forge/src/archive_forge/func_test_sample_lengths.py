import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
import pandas.core.common as com
def test_sample_lengths(self, obj):
    assert len(obj.sample(n=4) == 4)
    assert len(obj.sample(frac=0.34) == 3)
    assert len(obj.sample(frac=0.36) == 4)