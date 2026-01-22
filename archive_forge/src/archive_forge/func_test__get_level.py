from __future__ import print_function
import sys
import six
import numpy as np
from patsy import PatsyError
from patsy.util import (repr_pretty_delegate, repr_pretty_impl,
def test__get_level():
    assert _get_level(['a', 'b', 'c'], 0) == 0
    assert _get_level(['a', 'b', 'c'], -1) == 2
    assert _get_level(['a', 'b', 'c'], 'b') == 1
    assert _get_level([2, 1, 0], 0) == 2
    import pytest
    pytest.raises(PatsyError, _get_level, ['a', 'b'], 2)
    pytest.raises(PatsyError, _get_level, ['a', 'b'], -3)
    pytest.raises(PatsyError, _get_level, ['a', 'b'], 'c')
    if not six.PY3:
        assert _get_level(['a', 'b', 'c'], long(0)) == 0
        assert _get_level(['a', 'b', 'c'], long(-1)) == 2
        assert _get_level([2, 1, 0], long(0)) == 2