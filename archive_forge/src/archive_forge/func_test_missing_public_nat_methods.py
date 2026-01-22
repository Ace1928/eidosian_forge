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
@pytest.mark.parametrize('klass,expected', [(Timestamp, ['normalize', 'to_julian_date', 'to_period', 'unit']), (Timedelta, ['components', 'resolution_string', 'to_pytimedelta', 'to_timedelta64', 'unit', 'view'])])
def test_missing_public_nat_methods(klass, expected):
    nat_names = dir(NaT)
    klass_names = dir(klass)
    missing = [x for x in klass_names if x not in nat_names and (not x.startswith('_'))]
    missing.sort()
    assert missing == expected