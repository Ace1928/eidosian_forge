import sys
import gc
from hypothesis import given
from hypothesis.extra import numpy as hynp
import pytest
import numpy as np
from numpy.testing import (
from numpy.core.arrayprint import _typelessdata
import textwrap
@given(hynp.from_dtype(np.dtype('U')))
def test_any_text(self, text):
    a = np.array([text, text, text])
    assert_equal(a[0], text)
    expected_repr = '[{0!r} {0!r}\n {0!r}]'.format(text)
    result = np.array2string(a, max_line_width=len(repr(text)) * 2 + 3)
    assert_equal(result, expected_repr)