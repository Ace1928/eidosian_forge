from __future__ import print_function, division, absolute_import
from datetime import datetime
from decimal import Decimal
import pytest
from petl.test.helpers import eq_, ieq
from petl.comparison import Comparable
def test_comparable_datetime():
    dt = datetime.now().replace
    d = [dt(hour=12), dt(hour=3), dt(hour=1)]
    a = sorted(d, key=Comparable)
    e = [dt(hour=1), dt(hour=3), dt(hour=12)]
    eq_(e, a)
    d = [dt(hour=12), None, dt(hour=3), dt(hour=1)]
    a = sorted(d, key=Comparable)
    e = [None, dt(hour=1), dt(hour=3), dt(hour=12)]
    eq_(e, a)
    d = [dt(hour=12), None, dt(hour=3), u'b', True, b'ccc', False, b'aa', -1, 3.4]
    a = sorted(d, key=Comparable)
    e = [None, -1, False, True, 3.4, dt(hour=3), dt(hour=12), b'aa', b'ccc', u'b']
    eq_(e, a)