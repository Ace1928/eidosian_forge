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
def test_get_ipython_dir_7():
    """test_get_ipython_dir_7, test home directory expansion on IPYTHONDIR"""
    home_dir = os.path.normpath(os.path.expanduser('~'))
    with modified_env({'IPYTHONDIR': os.path.join('~', 'somewhere')}), patch.object(paths, '_writable_dir', return_value=True):
        ipdir = paths.get_ipython_dir()
    assert ipdir == os.path.join(home_dir, 'somewhere')