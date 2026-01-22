import contextlib
import sys
import warnings
import itertools
import operator
import platform
from numpy._utils import _pep440
import pytest
from hypothesis import given, settings
from hypothesis.strategies import sampled_from
from hypothesis.extra import numpy as hynp
import numpy as np
from numpy.testing import (
def test_numpy_scalar_relational_operators(self):
    for dt1 in np.typecodes['AllInteger']:
        assert_(1 > np.array(0, dtype=dt1)[()], 'type %s failed' % (dt1,))
        assert_(not 1 < np.array(0, dtype=dt1)[()], 'type %s failed' % (dt1,))
        for dt2 in np.typecodes['AllInteger']:
            assert_(np.array(1, dtype=dt1)[()] > np.array(0, dtype=dt2)[()], 'type %s and %s failed' % (dt1, dt2))
            assert_(not np.array(1, dtype=dt1)[()] < np.array(0, dtype=dt2)[()], 'type %s and %s failed' % (dt1, dt2))
    for dt1 in 'BHILQP':
        assert_(-1 < np.array(1, dtype=dt1)[()], 'type %s failed' % (dt1,))
        assert_(not -1 > np.array(1, dtype=dt1)[()], 'type %s failed' % (dt1,))
        assert_(-1 != np.array(1, dtype=dt1)[()], 'type %s failed' % (dt1,))
        for dt2 in 'bhilqp':
            assert_(np.array(1, dtype=dt1)[()] > np.array(-1, dtype=dt2)[()], 'type %s and %s failed' % (dt1, dt2))
            assert_(not np.array(1, dtype=dt1)[()] < np.array(-1, dtype=dt2)[()], 'type %s and %s failed' % (dt1, dt2))
            assert_(np.array(1, dtype=dt1)[()] != np.array(-1, dtype=dt2)[()], 'type %s and %s failed' % (dt1, dt2))
    for dt1 in 'bhlqp' + np.typecodes['Float']:
        assert_(1 > np.array(-1, dtype=dt1)[()], 'type %s failed' % (dt1,))
        assert_(not 1 < np.array(-1, dtype=dt1)[()], 'type %s failed' % (dt1,))
        assert_(-1 == np.array(-1, dtype=dt1)[()], 'type %s failed' % (dt1,))
        for dt2 in 'bhlqp' + np.typecodes['Float']:
            assert_(np.array(1, dtype=dt1)[()] > np.array(-1, dtype=dt2)[()], 'type %s and %s failed' % (dt1, dt2))
            assert_(not np.array(1, dtype=dt1)[()] < np.array(-1, dtype=dt2)[()], 'type %s and %s failed' % (dt1, dt2))
            assert_(np.array(-1, dtype=dt1)[()] == np.array(-1, dtype=dt2)[()], 'type %s and %s failed' % (dt1, dt2))