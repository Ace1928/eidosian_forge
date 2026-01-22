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
def test_stub_loading(tmp_path):
    stub = tmp_path / 'stub.pyi'
    stub.write_text(FAKE_STUB)
    _get, _dir, _all = lazy.attach_stub('my_module', str(stub))
    expect = {'gaussian', 'sobel', 'scharr', 'prewitt', 'roberts', 'rank'}
    assert set(_dir()) == set(_all) == expect