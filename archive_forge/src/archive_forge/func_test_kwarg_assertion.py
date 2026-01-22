import re
import numpy as np
import pytest
from pandas._libs.tslibs.timedeltas import (
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('kwargs', [{'Seconds': 1}, {'seconds': 1, 'Nanoseconds': 1}, {'Foo': 2}])
def test_kwarg_assertion(kwargs):
    err_message = 'cannot construct a Timedelta from the passed arguments, allowed keywords are [weeks, days, hours, minutes, seconds, milliseconds, microseconds, nanoseconds]'
    with pytest.raises(ValueError, match=re.escape(err_message)):
        Timedelta(**kwargs)