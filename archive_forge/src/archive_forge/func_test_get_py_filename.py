import os
import shutil
import sys
import tempfile
import unittest
from contextlib import contextmanager
from importlib import reload
from os.path import abspath, join
from unittest.mock import patch
import pytest
from tempfile import TemporaryDirectory
import IPython
from IPython import paths
from IPython.testing import decorators as dec
from IPython.testing.decorators import (
from IPython.testing.tools import make_tempfile
from IPython.utils import path
@with_environment
def test_get_py_filename():
    os.chdir(TMP_TEST_DIR)
    with make_tempfile('foo.py'):
        assert path.get_py_filename('foo.py') == 'foo.py'
        assert path.get_py_filename('foo') == 'foo.py'
    with make_tempfile('foo'):
        assert path.get_py_filename('foo') == 'foo'
        pytest.raises(IOError, path.get_py_filename, 'foo.py')
    pytest.raises(IOError, path.get_py_filename, 'foo')
    pytest.raises(IOError, path.get_py_filename, 'foo.py')
    true_fn = 'foo with spaces.py'
    with make_tempfile(true_fn):
        assert path.get_py_filename('foo with spaces') == true_fn
        assert path.get_py_filename('foo with spaces.py') == true_fn
        pytest.raises(IOError, path.get_py_filename, '"foo with spaces.py"')
        pytest.raises(IOError, path.get_py_filename, "'foo with spaces.py'")