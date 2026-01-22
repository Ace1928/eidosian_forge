from datetime import (
import operator
import numpy as np
import pytest
import pytz
from pandas._libs.tslibs import iNaT
from pandas.compat.numpy import np_version_gte1p24p3
from pandas import (
import pandas._testing as tm
from pandas.core import roperator
from pandas.core.arrays import (
@pytest.mark.parametrize('method', ['astimezone', 'combine', 'ctime', 'dst', 'fromordinal', 'fromtimestamp', 'fromisocalendar', 'isocalendar', 'strftime', 'strptime', 'time', 'timestamp', 'timetuple', 'timetz', 'toordinal', 'tzname', 'utcfromtimestamp', 'utcnow', 'utcoffset', 'utctimetuple', 'timestamp'])
def test_nat_methods_raise(method):
    msg = f'NaTType does not support {method}'
    with pytest.raises(ValueError, match=msg):
        getattr(NaT, method)()