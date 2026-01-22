import sys
import pytest
import textwrap
import subprocess
import numpy as np
import numpy.core._multiarray_tests as _multiarray_tests
from numpy import array, arange, nditer, all
from numpy.testing import (
def test_iter_flags_errors():
    a = arange(6)
    assert_raises(ValueError, nditer, [], [], [])
    assert_raises(ValueError, nditer, [a] * 100, [], [['readonly']] * 100)
    assert_raises(ValueError, nditer, [a], ['bad flag'], [['readonly']])
    assert_raises(ValueError, nditer, [a], [], [['readonly', 'bad flag']])
    assert_raises(ValueError, nditer, [a], [], [['readonly']], order='G')
    assert_raises(ValueError, nditer, [a], [], [['readonly']], casting='noon')
    assert_raises(ValueError, nditer, [a] * 3, [], [['readonly']] * 2)
    assert_raises(ValueError, nditer, a, ['c_index', 'f_index'], [['readonly']])
    assert_raises(ValueError, nditer, a, ['external_loop', 'multi_index'], [['readonly']])
    assert_raises(ValueError, nditer, a, ['external_loop', 'c_index'], [['readonly']])
    assert_raises(ValueError, nditer, a, ['external_loop', 'f_index'], [['readonly']])
    assert_raises(ValueError, nditer, a, [], [[]])
    assert_raises(ValueError, nditer, a, [], [['readonly', 'writeonly']])
    assert_raises(ValueError, nditer, a, [], [['readonly', 'readwrite']])
    assert_raises(ValueError, nditer, a, [], [['writeonly', 'readwrite']])
    assert_raises(ValueError, nditer, a, [], [['readonly', 'writeonly', 'readwrite']])
    assert_raises(TypeError, nditer, 1.5, [], [['writeonly']])
    assert_raises(TypeError, nditer, 1.5, [], [['readwrite']])
    assert_raises(TypeError, nditer, np.int32(1), [], [['writeonly']])
    assert_raises(TypeError, nditer, np.int32(1), [], [['readwrite']])
    a.flags.writeable = False
    assert_raises(ValueError, nditer, a, [], [['writeonly']])
    assert_raises(ValueError, nditer, a, [], [['readwrite']])
    a.flags.writeable = True
    i = nditer(arange(6), [], [['readonly']])
    assert_raises(ValueError, lambda i: i.multi_index, i)
    assert_raises(ValueError, lambda i: i.index, i)

    def assign_multi_index(i):
        i.multi_index = (0,)

    def assign_index(i):
        i.index = 0

    def assign_iterindex(i):
        i.iterindex = 0

    def assign_iterrange(i):
        i.iterrange = (0, 1)
    i = nditer(arange(6), ['external_loop'])
    assert_raises(ValueError, assign_multi_index, i)
    assert_raises(ValueError, assign_index, i)
    assert_raises(ValueError, assign_iterindex, i)
    assert_raises(ValueError, assign_iterrange, i)
    i = nditer(arange(6), ['buffered'])
    assert_raises(ValueError, assign_multi_index, i)
    assert_raises(ValueError, assign_index, i)
    assert_raises(ValueError, assign_iterrange, i)
    assert_raises(ValueError, nditer, np.array([]))