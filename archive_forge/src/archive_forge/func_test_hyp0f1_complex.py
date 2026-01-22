import numpy as np
from numpy.testing import assert_, assert_allclose
from numpy import pi
import pytest
import itertools
from scipy._lib import _pep440
import scipy.special as sc
from scipy.special._testutils import (
from scipy.special._mptestutils import (
from scipy.special._ufuncs import (
def test_hyp0f1_complex(self):
    assert_mpmath_equal(lambda a, z: sc.hyp0f1(a.real, z), exception_to_nan(lambda a, x: mpmath.hyp0f1(a, x, **HYPERKW)), [Arg(-10, 10), ComplexArg(complex(-120, -120), complex(120, 120))])