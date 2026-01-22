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
def test_get_ipython_cache_dir():
    with modified_env({'HOME': HOME_TEST_DIR}):
        if os.name == 'posix':
            os.makedirs(os.path.join(HOME_TEST_DIR, '.cache'))
            with modified_env({'XDG_CACHE_HOME': None}):
                ipdir = paths.get_ipython_cache_dir()
            assert os.path.join(HOME_TEST_DIR, '.cache', 'ipython') == ipdir
            assert_isdir(ipdir)
            with modified_env({'XDG_CACHE_HOME': XDG_CACHE_DIR}):
                ipdir = paths.get_ipython_cache_dir()
            assert_isdir(ipdir)
            assert ipdir == os.path.join(XDG_CACHE_DIR, 'ipython')
        else:
            assert paths.get_ipython_cache_dir() == paths.get_ipython_dir()