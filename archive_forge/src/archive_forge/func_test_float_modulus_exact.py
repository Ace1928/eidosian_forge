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
def test_float_modulus_exact(self):
    nlst = list(range(-127, 0))
    plst = list(range(1, 128))
    dividend = nlst + [0] + plst
    divisor = nlst + plst
    arg = list(itertools.product(dividend, divisor))
    tgt = list((divmod(*t) for t in arg))
    a, b = np.array(arg, dtype=int).T
    tgtdiv, tgtrem = np.array(tgt, dtype=float).T
    tgtdiv = np.where((tgtdiv == 0.0) & ((b < 0) ^ (a < 0)), -0.0, tgtdiv)
    tgtrem = np.where((tgtrem == 0.0) & (b < 0), -0.0, tgtrem)
    for op in [floordiv_and_mod, divmod]:
        for dt in np.typecodes['Float']:
            msg = 'op: %s, dtype: %s' % (op.__name__, dt)
            fa = a.astype(dt)
            fb = b.astype(dt)
            div, rem = zip(*[op(a_, b_) for a_, b_ in zip(fa, fb)])
            assert_equal(div, tgtdiv, err_msg=msg)
            assert_equal(rem, tgtrem, err_msg=msg)