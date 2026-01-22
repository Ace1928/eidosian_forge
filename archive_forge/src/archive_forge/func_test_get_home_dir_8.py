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
@skip_if_not_win32
@with_environment
def test_get_home_dir_8():
    """Using registry hack for 'My Documents', os=='nt'

    HOMESHARE, HOMEDRIVE, HOMEPATH, USERPROFILE and others are missing.
    """
    os.name = 'nt'
    for key in ['HOME', 'HOMESHARE', 'HOMEDRIVE', 'HOMEPATH', 'USERPROFILE']:
        env.pop(key, None)

    class key:

        def __enter__(self):
            pass

        def Close(self):
            pass

        def __exit__(*args, **kwargs):
            pass
    with patch.object(wreg, 'OpenKey', return_value=key()), patch.object(wreg, 'QueryValueEx', return_value=[abspath(HOME_TEST_DIR)]):
        home_dir = path.get_home_dir()
    assert home_dir == abspath(HOME_TEST_DIR)