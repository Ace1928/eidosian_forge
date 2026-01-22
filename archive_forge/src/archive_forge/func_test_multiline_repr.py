import sys
import gc
from hypothesis import given
from hypothesis.extra import numpy as hynp
import pytest
import numpy as np
from numpy.testing import (
from numpy.core.arrayprint import _typelessdata
import textwrap
def test_multiline_repr(self):

    class MultiLine:

        def __repr__(self):
            return 'Line 1\nLine 2'
    a = np.array([[None, MultiLine()], [MultiLine(), None]])
    assert_equal(np.array2string(a), '[[None Line 1\n       Line 2]\n [Line 1\n  Line 2 None]]')
    assert_equal(np.array2string(a, max_line_width=5), '[[None\n  Line 1\n  Line 2]\n [Line 1\n  Line 2\n  None]]')
    assert_equal(repr(a), 'array([[None, Line 1\n              Line 2],\n       [Line 1\n        Line 2, None]], dtype=object)')

    class MultiLineLong:

        def __repr__(self):
            return 'Line 1\nLooooooooooongestLine2\nLongerLine 3'
    a = np.array([[None, MultiLineLong()], [MultiLineLong(), None]])
    assert_equal(repr(a), 'array([[None, Line 1\n              LooooooooooongestLine2\n              LongerLine 3          ],\n       [Line 1\n        LooooooooooongestLine2\n        LongerLine 3          , None]], dtype=object)')
    assert_equal(np.array_repr(a, 20), 'array([[None,\n        Line 1\n        LooooooooooongestLine2\n        LongerLine 3          ],\n       [Line 1\n        LooooooooooongestLine2\n        LongerLine 3          ,\n        None]],\n      dtype=object)')