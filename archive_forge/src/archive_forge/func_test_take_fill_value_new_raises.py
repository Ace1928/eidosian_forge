import numpy as np
import pytest
from pandas import Categorical
import pandas._testing as tm
def test_take_fill_value_new_raises(self):
    cat = Categorical(['a', 'b', 'c'])
    xpr = 'Cannot setitem on a Categorical with a new category \\(d\\)'
    with pytest.raises(TypeError, match=xpr):
        cat.take([0, 1, -1], fill_value='d', allow_fill=True)