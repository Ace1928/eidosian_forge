import os
import sys
import pytest
import numpy as np
from . import util
@pytest.mark.skipif(sys.platform == 'win32', reason='Fails with MinGW64 Gfortran (Issue #9673)')
def test_common_gh19161(self):
    assert self.module.data.x == 0