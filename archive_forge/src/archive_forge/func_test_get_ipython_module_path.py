import errno
import os
import shutil
import tempfile
import warnings
from unittest.mock import patch
from tempfile import TemporaryDirectory
from testpath import assert_isdir, assert_isfile, modified_env
from IPython import paths
from IPython.testing.decorators import skip_win32
def test_get_ipython_module_path():
    ipapp_path = paths.get_ipython_module_path('IPython.terminal.ipapp')
    assert_isfile(ipapp_path)