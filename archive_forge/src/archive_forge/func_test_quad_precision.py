import os
import pytest
import platform
from numpy.f2py.crackfortran import (
from . import util
@pytest.mark.xfail(platform.machine().lower().startswith('ppc'), reason='Some PowerPC may not support full IEEE 754 precision')
def test_quad_precision(self):
    """
        Test kind_func for quadruple precision [`real(16)`] of 32+ digits .
        """
    selectedrealkind = self.module.selectedrealkind
    for i in range(32, 40):
        assert selectedrealkind(i) == selected_real_kind(i), f'selectedrealkind({i}): expected {selected_real_kind(i)!r} but got {selectedrealkind(i)!r}'