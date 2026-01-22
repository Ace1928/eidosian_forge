from symengine.test_utilities import raises
from symengine import (Symbol, sin, cos, Integer, Add, I, RealDouble, ComplexDouble, sqrt)
from unittest.case import SkipTest
def test_n_mpfr():
    x = sqrt(Integer(2))
    try:
        from symengine import RealMPFR
        y = RealMPFR('1.41421356237309504880169', 75)
        assert x.n(75, real=True) == y
    except ImportError:
        raises(ValueError, lambda: x.n(75, real=True))
        raises(ValueError, lambda: x.n(75))
        raise SkipTest('No MPFR support')