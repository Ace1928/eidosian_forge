import operator as op
from numbers import Number
import pytest
import numpy as np
from numpy.polynomial import (
from numpy.testing import (
from numpy.polynomial.polyutils import RankWarning
def test_bad_conditioned_fit(Poly):
    x = [0.0, 0.0, 1.0]
    y = [1.0, 2.0, 3.0]
    with pytest.warns(RankWarning) as record:
        Poly.fit(x, y, 2)
    assert record[0].message.args[0] == 'The fit may be poorly conditioned'