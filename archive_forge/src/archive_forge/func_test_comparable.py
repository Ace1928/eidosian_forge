from __future__ import print_function, division, absolute_import
from datetime import datetime
from decimal import Decimal
import pytest
from petl.test.helpers import eq_, ieq
from petl.comparison import Comparable
def test_comparable():
    d = [True, False]
    a = sorted(d, key=Comparable)
    e = [False, True]
    eq_(e, a)
    d = [3, 1, 2]
    a = sorted(d, key=Comparable)
    e = [1, 2, 3]
    eq_(e, a)
    d = [3.0, 1.2, 2.5]
    a = sorted(d, key=Comparable)
    e = [1.2, 2.5, 3.0]
    eq_(e, a)
    d = [3.0, 1, 2.5, Decimal('1.5')]
    a = sorted(d, key=Comparable)
    e = [1, Decimal('1.5'), 2.5, 3.0]
    eq_(e, a)
    d = [True, False, -1.2, 2, 0.5]
    a = sorted(d, key=Comparable)
    e = [-1.2, False, 0.5, True, 2]
    eq_(e, a)
    d = [3, None, 2.5]
    a = sorted(d, key=Comparable)
    e = [None, 2.5, 3.0]
    eq_(e, a)
    d = [b'b', b'ccc', b'aa']
    a = sorted(d, key=Comparable)
    e = [b'aa', b'b', b'ccc']
    eq_(e, a)
    d = [u'b', u'ccc', u'aa']
    a = sorted(d, key=Comparable)
    e = [u'aa', u'b', u'ccc']
    eq_(e, a)
    d = [u'b', b'ccc', b'aa']
    a = sorted(d, key=Comparable)
    e = [b'aa', b'ccc', u'b']
    eq_(e, a)
    d = [b'b', b'ccc', None, b'aa']
    a = sorted(d, key=Comparable)
    e = [None, b'aa', b'b', b'ccc']
    eq_(e, a)
    d = [u'b', u'ccc', None, u'aa']
    a = sorted(d, key=Comparable)
    e = [None, u'aa', u'b', u'ccc']
    eq_(e, a)
    d = [u'b', b'ccc', None, b'aa']
    a = sorted(d, key=Comparable)
    e = [None, b'aa', b'ccc', u'b']
    eq_(e, a)
    d = [u'b', True, b'ccc', False, None, b'aa', -1, 3.4]
    a = sorted(d, key=Comparable)
    e = [None, -1, False, True, 3.4, b'aa', b'ccc', u'b']
    eq_(e, a)