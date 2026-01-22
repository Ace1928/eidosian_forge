import code
import platform
import pytest
import sys
from tempfile import TemporaryFile
import numpy as np
from numpy.testing import assert_, assert_equal, assert_raises, IS_MUSL
def float64_vs_python(self):
    assert_equal(repr(np.float64(0.1)), repr(0.1))
    assert_(repr(np.float64(0.20000000000000004)) != repr(0.2))