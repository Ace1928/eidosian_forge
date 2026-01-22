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
@pytest.mark.parametrize('name', ['ModuleDeprecationWarning', 'VisibleDeprecationWarning', 'ComplexWarning', 'TooHardError', 'AxisError'])
def test_moved_exceptions(name):
    assert name in np.__all__
    assert name not in np.__dir__()
    assert getattr(np, name).__module__ == 'numpy.exceptions'
    assert name in np.exceptions.__all__
    getattr(np.exceptions, name)