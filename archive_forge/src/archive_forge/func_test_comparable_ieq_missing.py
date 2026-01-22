from __future__ import print_function, division, absolute_import
from datetime import datetime
from decimal import Decimal
import pytest
from petl.test.helpers import eq_, ieq
from petl.comparison import Comparable
def test_comparable_ieq_missing():
    x = ['a', 'b', 'c']
    y = ['a', 'b']
    with pytest.raises(AssertionError):
        ieq(x, y)
    with pytest.raises(AssertionError):
        ieq(y, x)