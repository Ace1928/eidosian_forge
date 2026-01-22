import numpy as np
import pytest
from pandas import (
from pandas.core.strings.accessor import StringMethods
def test_api_mi_raises():
    mi = MultiIndex.from_arrays([['a', 'b', 'c']])
    msg = 'Can only use .str accessor with Index, not MultiIndex'
    with pytest.raises(AttributeError, match=msg):
        mi.str
    assert not hasattr(mi, 'str')