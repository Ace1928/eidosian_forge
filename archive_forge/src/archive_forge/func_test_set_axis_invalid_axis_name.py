import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('axis', [3, 'foo'])
def test_set_axis_invalid_axis_name(self, axis, obj):
    with pytest.raises(ValueError, match='No axis named'):
        obj.set_axis(list('abc'), axis=axis)