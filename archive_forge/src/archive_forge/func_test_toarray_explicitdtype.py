from __future__ import absolute_import, print_function, division
import pytest
import petl as etl
from petl.test.helpers import ieq, eq_, assert_almost_equal
from petl.io.numpy import toarray, fromarray, torecarray
def test_toarray_explicitdtype():
    t = [('foo', 'bar', 'baz'), ('apples', 1, 2.5), ('oranges', 3, 4.4), ('pears', 7, 0.1)]
    a = toarray(t, dtype=[('A', 'U4'), ('B', 'i2'), ('C', 'f4')])
    assert isinstance(a, np.ndarray)
    assert isinstance(a['A'], np.ndarray)
    assert isinstance(a['B'], np.ndarray)
    assert isinstance(a['C'], np.ndarray)
    eq_('appl', a['A'][0])
    eq_('oran', a['A'][1])
    eq_('pear', a['A'][2])
    eq_(1, a['B'][0])
    eq_(3, a['B'][1])
    eq_(7, a['B'][2])
    assert_almost_equal(2.5, a['C'][0], places=6)
    assert_almost_equal(4.4, a['C'][1], places=6)
    assert_almost_equal(0.1, a['C'][2], places=6)