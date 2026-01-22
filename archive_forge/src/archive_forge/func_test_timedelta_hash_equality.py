from datetime import timedelta
import sys
from hypothesis import (
import numpy as np
import pytest
from pandas._libs import lib
from pandas._libs.tslibs import (
from pandas._libs.tslibs.dtypes import NpyDatetimeUnit
from pandas.errors import OutOfBoundsTimedelta
from pandas import (
import pandas._testing as tm
def test_timedelta_hash_equality(self):
    v = Timedelta(1, 'D')
    td = timedelta(days=1)
    assert hash(v) == hash(td)
    d = {td: 2}
    assert d[v] == 2
    tds = [Timedelta(seconds=1) + Timedelta(days=n) for n in range(20)]
    assert all((hash(td) == hash(td.to_pytimedelta()) for td in tds))
    ns_td = Timedelta(1, 'ns')
    assert hash(ns_td) != hash(ns_td.to_pytimedelta())