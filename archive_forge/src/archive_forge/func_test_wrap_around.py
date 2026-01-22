import numpy as np
from skimage.restoration import unwrap_phase
import sys
from skimage._shared import testing
from skimage._shared.testing import (
from skimage._shared._warnings import expected_warnings
@skipif(sys.version_info[:2] == (3, 4), reason="Doesn't work with python 3.4. See issue #3079")
@testing.parametrize('ndim, axis', dim_axis)
def test_wrap_around(ndim, axis):
    check_wrap_around(ndim, axis)