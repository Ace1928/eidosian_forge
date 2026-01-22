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
def test_get_ipython_dir_3():
    """test_get_ipython_dir_3, use XDG if defined and exists, and .ipython doesn't exist."""
    tmphome = TemporaryDirectory()
    try:
        with patch_get_home_dir(tmphome.name), patch('os.name', 'posix'), modified_env({'IPYTHON_DIR': None, 'IPYTHONDIR': None, 'XDG_CONFIG_HOME': XDG_TEST_DIR}), warnings.catch_warnings(record=True) as w:
            ipdir = paths.get_ipython_dir()
        assert ipdir == os.path.join(tmphome.name, XDG_TEST_DIR, 'ipython')
        assert len(w) == 0
    finally:
        tmphome.cleanup()