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
def test_parallel_load():
    pytest.importorskip('numpy')
    subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), 'import_np_parallel.py')])