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
def test_numpy_linalg():
    bad_results = check_dir(np.linalg)
    assert bad_results == {}