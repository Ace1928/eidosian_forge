import importlib
import os
import subprocess
import sys
import types
from unittest import mock
import pytest
import lazy_loader as lazy
from . import rank
from ._gaussian import gaussian
from .edges import sobel, scharr, prewitt, roberts
def test_lazy_import_impact_on_sys_modules():
    math = lazy.load('math')
    anything_not_real = lazy.load('anything_not_real')
    assert isinstance(math, types.ModuleType)
    assert 'math' in sys.modules
    assert isinstance(anything_not_real, lazy.DelayedImportErrorModule)
    assert 'anything_not_real' not in sys.modules
    pytest.importorskip('numpy')
    np = lazy.load('numpy')
    assert isinstance(np, types.ModuleType)
    assert 'numpy' in sys.modules
    np.pi
    assert isinstance(np, types.ModuleType)
    assert 'numpy' in sys.modules