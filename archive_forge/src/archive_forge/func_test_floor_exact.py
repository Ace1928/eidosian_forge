import sys
from contextlib import nullcontext
import numpy as np
import pytest
from packaging.version import Version
from ..casting import (
from ..testing import suppress_warnings
def test_floor_exact(max_digits):
    max_digits(4950)
    to_test = IEEE_floats + [float]
    try:
        type_info(np.longdouble)['nmant']
    except FloatingError:
        pass
    else:
        to_test.append(np.longdouble)
    int_flex = lambda x, t: int(floor_exact(x, t))
    int_ceex = lambda x, t: int(ceil_exact(x, t))
    for t in to_test:
        info = type_info(t)
        assert floor_exact(10 ** 4933, t) == np.inf
        assert ceil_exact(10 ** 4933, t) == np.inf
        assert floor_exact(-10 ** 4933, t) == -np.inf
        assert ceil_exact(-10 ** 4933, t) == -np.inf
        nmant = info['nmant']
        for i in range(nmant + 1):
            iv = 2 ** i
            for func in (int_flex, int_ceex):
                assert func(iv, t) == iv
                assert func(-iv, t) == -iv
                assert func(iv - 1, t) == iv - 1
                assert func(-iv + 1, t) == -iv + 1
        if t is np.longdouble and (on_powerpc() or longdouble_precision_improved()):
            continue
        iv = 2 ** (nmant + 1)
        assert int_flex(iv + 1, t) == iv
        assert int_ceex(iv + 1, t) == iv + 2
        assert int_flex(-iv - 1, t) == -iv - 2
        assert int_ceex(-iv - 1, t) == -iv
        for i in range(5):
            iv = 2 ** (nmant + 1 + i)
            gap = 2 ** (i + 1)
            assert int(t(iv) + t(gap)) == iv + gap
            for j in range(1, gap):
                assert int_flex(iv + j, t) == iv
                assert int_flex(iv + gap + j, t) == iv + gap
                assert int_ceex(iv + j, t) == iv + gap
                assert int_ceex(iv + gap + j, t) == iv + 2 * gap
            for j in range(1, gap):
                assert int_flex(-iv - j, t) == -iv - gap
                assert int_flex(-iv - gap - j, t) == -iv - 2 * gap
                assert int_ceex(-iv - j, t) == -iv
                assert int_ceex(-iv - gap - j, t) == -iv - gap