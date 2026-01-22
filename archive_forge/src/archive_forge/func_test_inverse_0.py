import numpy as np
import pytest
from numpy import pi
from numpy.testing import assert_array_almost_equal, assert_array_equal
from .. import eulerangles as nea
from .. import quaternions as nq
def test_inverse_0():
    iq = nq.inverse((1, 0, 0, 0))
    assert iq.dtype.kind == 'f'