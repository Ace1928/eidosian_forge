import sys
import sysconfig
import subprocess
import pkgutil
import types
import importlib
import warnings
import numpy as np
import numpy
import pytest
from numpy.testing import IS_WASM
def test_numpy_fft():
    bad_results = check_dir(np.fft)
    assert bad_results == {}