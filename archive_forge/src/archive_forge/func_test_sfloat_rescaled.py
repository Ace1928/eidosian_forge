import pytest
import numpy as np
from numpy.testing import assert_array_equal
from numpy.core._multiarray_umath import (
def test_sfloat_rescaled(self):
    sf = SF(1.0)
    sf2 = sf.scaled_by(2.0)
    assert sf2.get_scaling() == 2.0
    sf6 = sf2.scaled_by(3.0)
    assert sf6.get_scaling() == 6.0