import collections
from functools import partial
import string
import subprocess
import sys
import textwrap
import numpy as np
import pytest
import pandas as pd
from pandas import Series
import pandas._testing as tm
from pandas.core import ops
import pandas.core.common as com
from pandas.util.version import Version
@pytest.mark.parametrize('with_exception', [True, False])
def test_temp_setattr(with_exception):
    ser = Series(dtype=object)
    ser.name = 'first'
    match = 'Inside exception raised' if with_exception else 'Outside exception raised'
    with pytest.raises(ValueError, match=match):
        with com.temp_setattr(ser, 'name', 'second'):
            assert ser.name == 'second'
            if with_exception:
                raise ValueError('Inside exception raised')
        raise ValueError('Outside exception raised')
    assert ser.name == 'first'