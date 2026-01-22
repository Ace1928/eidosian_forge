import warnings
import platform
import pytest
import numpy as np
from numpy.testing import (
from numpy.core.tests._locales import CommaDecimalPointLocale
@pytest.mark.skipif(not (IS_MUSL and platform.machine() == 'x86_64'), reason='only need to run on musllinux_x86_64')
def test_musllinux_x86_64_signature():
    known_sigs = [b'\xcd\xcc\xcc\xcc\xcc\xcc\xcc\xcc\xfb\xbf']
    sig = (np.longdouble(-1.0) / np.longdouble(10.0)).newbyteorder('<').tobytes()[:10]
    assert sig in known_sigs