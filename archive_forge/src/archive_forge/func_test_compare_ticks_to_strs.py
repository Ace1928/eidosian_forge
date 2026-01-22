from datetime import (
from hypothesis import (
import numpy as np
import pytest
from pandas._libs.tslibs.offsets import delta_to_tick
from pandas.errors import OutOfBoundsTimedelta
from pandas import (
import pandas._testing as tm
from pandas._testing._hypothesis import INT_NEG_999_TO_POS_999
from pandas.tests.tseries.offsets.common import assert_offset_equal
from pandas.tseries import offsets
from pandas.tseries.offsets import (
@pytest.mark.parametrize('cls', tick_classes)
def test_compare_ticks_to_strs(cls):
    off = cls(19)
    assert not off == 'infer'
    assert not 'foo' == off
    instance_type = '.'.join([cls.__module__, cls.__name__])
    msg = f"'<'|'<='|'>'|'>=' not supported between instances of 'str' and '{instance_type}'|'{instance_type}' and 'str'"
    for left, right in [('infer', off), (off, 'infer')]:
        with pytest.raises(TypeError, match=msg):
            left < right
        with pytest.raises(TypeError, match=msg):
            left <= right
        with pytest.raises(TypeError, match=msg):
            left > right
        with pytest.raises(TypeError, match=msg):
            left >= right