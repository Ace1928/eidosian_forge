import numpy as np
import pytest
from pandas.core.dtypes.common import ensure_platform_int
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('start,stop,step', [('foo', 'bar', 'baz'), ('0', '1', '2')])
def test_cant_or_shouldnt_cast(self, start, stop, step):
    msg = f'Wrong type {type(start)} for value {start}'
    with pytest.raises(TypeError, match=msg):
        RangeIndex(start, stop, step)