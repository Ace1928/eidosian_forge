import sys
import gc
from hypothesis import given
from hypothesis.extra import numpy as hynp
import pytest
import numpy as np
from numpy.testing import (
from numpy.core.arrayprint import _typelessdata
import textwrap
def test_legacy_mode_scalars(self):
    np.set_printoptions(legacy='1.13')
    assert_equal(str(np.float64(1.1234567891234568)), '1.12345678912')
    assert_equal(str(np.complex128(complex(1, np.nan))), '(1+nan*j)')
    np.set_printoptions(legacy=False)
    assert_equal(str(np.float64(1.1234567891234568)), '1.1234567891234568')
    assert_equal(str(np.complex128(complex(1, np.nan))), '(1+nanj)')