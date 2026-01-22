import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_set_axis_setattr_index_not_collection(self, obj):
    msg = 'Index\\(\\.\\.\\.\\) must be called with a collection of some kind, None was passed'
    with pytest.raises(TypeError, match=msg):
        obj.index = None