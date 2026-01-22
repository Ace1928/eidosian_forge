import os
import shutil
import subprocess
import sys
import pytest
import numpy as np
from numpy.testing import IS_WASM
def test_get_timedelta64_value(install_temp):
    import checks
    td64 = np.timedelta64(12345, 'h')
    result = checks.get_td64_value(td64)
    expected = td64.view('i8')
    assert result == expected