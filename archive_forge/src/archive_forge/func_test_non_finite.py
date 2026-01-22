from math import nan, inf
import pytest
from numpy.core import array, arange, printoptions
import numpy.polynomial as poly
from numpy.testing import assert_equal, assert_
from fractions import Fraction
from decimal import Decimal
def test_non_finite(self):
    p = poly.Polynomial([nan, inf])
    assert str(p) == 'nan + inf x'
    assert p._repr_latex_() == '$x \\mapsto \\text{nan} + \\text{inf}\\,x$'
    with printoptions(nanstr='NAN', infstr='INF'):
        assert str(p) == 'NAN + INF x'
        assert p._repr_latex_() == '$x \\mapsto \\text{NAN} + \\text{INF}\\,x$'