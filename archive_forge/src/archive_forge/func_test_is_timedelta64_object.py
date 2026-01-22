import os
import shutil
import subprocess
import sys
import pytest
import numpy as np
from numpy.testing import IS_WASM
def test_is_timedelta64_object(install_temp):
    import checks
    assert checks.is_td64(np.timedelta64(1234))
    assert checks.is_td64(np.timedelta64(1234, 'ns'))
    assert checks.is_td64(np.timedelta64('NaT', 'ns'))
    assert not checks.is_td64(1)
    assert not checks.is_td64(None)
    assert not checks.is_td64('foo')
    assert not checks.is_td64(np.datetime64('now', 's'))