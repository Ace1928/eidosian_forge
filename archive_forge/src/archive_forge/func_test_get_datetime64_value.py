import os
import shutil
import subprocess
import sys
import pytest
import numpy as np
from numpy.testing import IS_WASM
def test_get_datetime64_value(install_temp):
    import checks
    dt64 = np.datetime64('2016-01-01', 'ns')
    result = checks.get_dt64_value(dt64)
    expected = dt64.view('i8')
    assert result == expected