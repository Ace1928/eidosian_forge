import os
import sys
import shutil
import tempfile
import unittest
import sysconfig
from copy import deepcopy
from test.support import os_helper
from distutils import log
from distutils.log import DEBUG, INFO, WARN, ERROR, FATAL
from distutils.core import Distribution
def copy_xxmodule_c(directory):
    """Helper for tests that need the xxmodule.c source file.

    Example use:

        def test_compile(self):
            copy_xxmodule_c(self.tmpdir)
            self.assertIn('xxmodule.c', os.listdir(self.tmpdir))

    If the source file can be found, it will be copied to *directory*.  If not,
    the test will be skipped.  Errors during copy are not caught.
    """
    filename = _get_xxmodule_path()
    if filename is None:
        raise unittest.SkipTest('cannot find xxmodule.c (test must run in the python build dir)')
    shutil.copy(filename, directory)