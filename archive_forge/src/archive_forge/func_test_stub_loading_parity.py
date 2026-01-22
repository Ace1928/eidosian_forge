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
def test_stub_loading_parity():
    from lazy_loader.tests import fake_pkg
    from_stub = lazy.attach_stub(fake_pkg.__name__, fake_pkg.__file__)
    stub_getter, stub_dir, stub_all = from_stub
    assert stub_all == fake_pkg.__all__
    assert stub_dir() == fake_pkg.__lazy_dir__()
    assert stub_getter('some_func') == fake_pkg.some_func