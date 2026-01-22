import os
import shutil
import pytest
from tempfile import mkstemp, mkdtemp
from subprocess import Popen, PIPE
import importlib.metadata
from distutils.errors import DistutilsError
from numpy.testing import assert_, assert_equal, assert_raises
from numpy.distutils import ccompiler, customized_ccompiler
from numpy.distutils.system_info import system_info, ConfigParser, mkl_info
from numpy.distutils.system_info import AliasedOptionError
from numpy.distutils.system_info import default_lib_dirs, default_include_dirs
from numpy.distutils import _shell_utils
def test_duplicate_options(self):
    tsi = self.c_dup_options
    assert_raises(AliasedOptionError, tsi.get_option_single, 'mylib_libs', 'libraries')
    assert_equal(tsi.get_libs('mylib_libs', [self._lib1]), [self._lib1])
    assert_equal(tsi.get_libs('libraries', [self._lib2]), [self._lib2])