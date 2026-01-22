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
def test_lazy_import_nonbuiltins():
    np = lazy.load('numpy')
    sp = lazy.load('scipy')
    if not isinstance(np, lazy.DelayedImportErrorModule):
        assert np.sin(np.pi) == pytest.approx(0, 1e-06)
    if isinstance(sp, lazy.DelayedImportErrorModule):
        try:
            sp.pi
            raise AssertionError()
        except ModuleNotFoundError:
            pass