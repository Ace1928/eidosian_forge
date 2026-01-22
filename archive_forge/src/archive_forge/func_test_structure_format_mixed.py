import sys
import gc
from hypothesis import given
from hypothesis.extra import numpy as hynp
import pytest
import numpy as np
from numpy.testing import (
from numpy.core.arrayprint import _typelessdata
import textwrap
def test_structure_format_mixed(self):
    dt = np.dtype([('name', np.str_, 16), ('grades', np.float64, (2,))])
    x = np.array([('Sarah', (8.0, 7.0)), ('John', (6.0, 7.0))], dtype=dt)
    assert_equal(np.array2string(x), "[('Sarah', [8., 7.]) ('John', [6., 7.])]")
    np.set_printoptions(legacy='1.13')
    try:
        A = np.zeros(shape=10, dtype=[('A', 'M8[s]')])
        A[5:].fill(np.datetime64('NaT'))
        assert_equal(np.array2string(A), textwrap.dedent("                [('1970-01-01T00:00:00',) ('1970-01-01T00:00:00',) ('1970-01-01T00:00:00',)\n                 ('1970-01-01T00:00:00',) ('1970-01-01T00:00:00',) ('NaT',) ('NaT',)\n                 ('NaT',) ('NaT',) ('NaT',)]"))
    finally:
        np.set_printoptions(legacy=False)
    assert_equal(np.array2string(A), textwrap.dedent("            [('1970-01-01T00:00:00',) ('1970-01-01T00:00:00',)\n             ('1970-01-01T00:00:00',) ('1970-01-01T00:00:00',)\n             ('1970-01-01T00:00:00',) (                'NaT',)\n             (                'NaT',) (                'NaT',)\n             (                'NaT',) (                'NaT',)]"))
    A = np.full(10, 123456, dtype=[('A', 'm8[s]')])
    A[5:].fill(np.datetime64('NaT'))
    assert_equal(np.array2string(A), textwrap.dedent("            [(123456,) (123456,) (123456,) (123456,) (123456,) ( 'NaT',) ( 'NaT',)\n             ( 'NaT',) ( 'NaT',) ( 'NaT',)]"))