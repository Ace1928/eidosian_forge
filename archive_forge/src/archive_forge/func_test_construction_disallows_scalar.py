import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_construction_disallows_scalar(self):
    msg = 'must be called with a collection of some kind'
    with pytest.raises(TypeError, match=msg):
        CategoricalIndex(data=1, categories=list('abcd'), ordered=False)
    with pytest.raises(TypeError, match=msg):
        CategoricalIndex(categories=list('abcd'), ordered=False)