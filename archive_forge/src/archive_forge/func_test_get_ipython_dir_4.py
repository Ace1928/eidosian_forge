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
def test_get_ipython_dir_4():
    """test_get_ipython_dir_4, warn if XDG and home both exist."""
    with patch_get_home_dir(HOME_TEST_DIR), patch('os.name', 'posix'):
        try:
            os.mkdir(os.path.join(XDG_TEST_DIR, 'ipython'))
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
        with modified_env({'IPYTHON_DIR': None, 'IPYTHONDIR': None, 'XDG_CONFIG_HOME': XDG_TEST_DIR}), warnings.catch_warnings(record=True) as w:
            ipdir = paths.get_ipython_dir()
        assert len(w) == 1
        assert 'Ignoring' in str(w[0])