import code
import platform
import pytest
import sys
from tempfile import TemporaryFile
import numpy as np
from numpy.testing import assert_, assert_equal, assert_raises, IS_MUSL
def test_dragon4_interface(self):
    tps = [np.float16, np.float32, np.float64]
    if hasattr(np, 'float128') and (not IS_MUSL):
        tps.append(np.float128)
    fpos = np.format_float_positional
    fsci = np.format_float_scientific
    for tp in tps:
        assert_equal(fpos(tp('1.0'), pad_left=4, pad_right=4), '   1.    ')
        assert_equal(fpos(tp('-1.0'), pad_left=4, pad_right=4), '  -1.    ')
        assert_equal(fpos(tp('-10.2'), pad_left=4, pad_right=4), ' -10.2   ')
        assert_equal(fsci(tp('1.23e1'), exp_digits=5), '1.23e+00001')
        assert_equal(fpos(tp('1.0'), unique=False, precision=4), '1.0000')
        assert_equal(fsci(tp('1.0'), unique=False, precision=4), '1.0000e+00')
        assert_equal(fpos(tp('1.'), unique=False, precision=4, trim='k'), '1.0000')
        assert_equal(fpos(tp('1.'), unique=False, precision=4, trim='.'), '1.')
        assert_equal(fpos(tp('1.2'), unique=False, precision=4, trim='.'), '1.2' if tp != np.float16 else '1.2002')
        assert_equal(fpos(tp('1.'), unique=False, precision=4, trim='0'), '1.0')
        assert_equal(fpos(tp('1.2'), unique=False, precision=4, trim='0'), '1.2' if tp != np.float16 else '1.2002')
        assert_equal(fpos(tp('1.'), trim='0'), '1.0')
        assert_equal(fpos(tp('1.'), unique=False, precision=4, trim='-'), '1')
        assert_equal(fpos(tp('1.2'), unique=False, precision=4, trim='-'), '1.2' if tp != np.float16 else '1.2002')
        assert_equal(fpos(tp('1.'), trim='-'), '1')
        assert_equal(fpos(tp('1.001'), precision=1, trim='-'), '1')