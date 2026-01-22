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
def patch_get_home_dir(dirpath):
    return patch.object(paths, 'get_home_dir', return_value=dirpath)