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
def test_lazy_import_subpackages():
    with pytest.warns(RuntimeWarning):
        hp = lazy.load('html.parser')
    assert 'html' in sys.modules
    assert type(sys.modules['html']) == type(pytest)
    assert isinstance(hp, importlib.util._LazyModule)
    assert 'html.parser' in sys.modules
    assert sys.modules['html.parser'] == hp