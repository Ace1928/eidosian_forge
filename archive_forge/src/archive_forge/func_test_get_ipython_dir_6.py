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
def test_get_ipython_dir_6():
    """test_get_ipython_dir_6, use home over XDG if defined and neither exist."""
    xdg = os.path.join(HOME_TEST_DIR, 'somexdg')
    os.mkdir(xdg)
    shutil.rmtree(os.path.join(HOME_TEST_DIR, '.ipython'))
    print(paths._writable_dir)
    with patch_get_home_dir(HOME_TEST_DIR), patch.object(paths, 'get_xdg_dir', return_value=xdg), patch('os.name', 'posix'), modified_env({'IPYTHON_DIR': None, 'IPYTHONDIR': None, 'XDG_CONFIG_HOME': None}), warnings.catch_warnings(record=True) as w:
        ipdir = paths.get_ipython_dir()
    assert ipdir == os.path.join(HOME_TEST_DIR, '.ipython')
    assert len(w) == 0