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
@pytest.mark.xfail(HAS_MKL, reason="`[DEFAULT]` override doesn't work if numpy is built with MKL support")
def test_overrides(self):
    previousDir = os.getcwd()
    cfg = os.path.join(self._dir1, 'site.cfg')
    shutil.copy(self._sitecfg, cfg)
    try:
        os.chdir(self._dir1)
        info = mkl_info()
        lib_dirs = info.cp['ALL']['library_dirs'].split(os.pathsep)
        assert info.get_lib_dirs() != lib_dirs
        with open(cfg) as fid:
            mkl = fid.read().replace('[ALL]', '[mkl]', 1)
        with open(cfg, 'w') as fid:
            fid.write(mkl)
        info = mkl_info()
        assert info.get_lib_dirs() == lib_dirs
        with open(cfg) as fid:
            dflt = fid.read().replace('[mkl]', '[DEFAULT]', 1)
        with open(cfg, 'w') as fid:
            fid.write(dflt)
        info = mkl_info()
        assert info.get_lib_dirs() == lib_dirs
    finally:
        os.chdir(previousDir)