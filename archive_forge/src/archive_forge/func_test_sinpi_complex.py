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
def test_sinpi_complex(self):
    assert_mpmath_equal(_sinpi, mpmath.sinpi, [ComplexArg()], nan_ok=False, rtol=2e-14)