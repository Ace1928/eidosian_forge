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
def test_float_modulus_roundoff(self):
    dt = np.typecodes['Float']
    for op in [floordiv_and_mod, divmod]:
        for dt1, dt2 in itertools.product(dt, dt):
            for sg1, sg2 in itertools.product((+1, -1), (+1, -1)):
                fmt = 'op: %s, dt1: %s, dt2: %s, sg1: %s, sg2: %s'
                msg = fmt % (op.__name__, dt1, dt2, sg1, sg2)
                a = np.array(sg1 * 78 * 6e-08, dtype=dt1)[()]
                b = np.array(sg2 * 6e-08, dtype=dt2)[()]
                div, rem = op(a, b)
                assert_equal(div * b + rem, a, err_msg=msg)
                if sg2 == -1:
                    assert_(b < rem <= 0, msg)
                else:
                    assert_(b > rem >= 0, msg)