import ctypes
import logging
import os
import platform
import shutil
import stat
import sys
import tempfile
import subprocess
from io import StringIO
import pyomo.common.unittest as unittest
import pyomo.common.envvar as envvar
from pyomo.common.log import LoggingIntercept
from pyomo.common.fileutils import (
from pyomo.common.download import FileDownloader
def test_PathManager(self):
    Executable = PathManager(find_executable, ExecutableData)
    self.tmpdir = os.path.abspath(tempfile.mkdtemp())
    envvar.PYOMO_CONFIG_DIR = self.tmpdir
    config_bindir = os.path.join(self.tmpdir, 'bin')
    os.mkdir(config_bindir)
    pathdir_name = 'in_path'
    pathdir = os.path.join(self.tmpdir, pathdir_name)
    os.mkdir(pathdir)
    os.environ['PATH'] = os.pathsep + pathdir + os.pathsep
    f_in_tmp = 'f_in_tmp'
    self._make_exec(os.path.join(self.tmpdir, f_in_tmp))
    f_in_path = 'f_in_path'
    self._make_exec(os.path.join(pathdir, f_in_path))
    f_in_cfg = 'f_in_configbin'
    self._make_exec(os.path.join(config_bindir, f_in_cfg))
    self.assertTrue(Executable(f_in_path).available())
    if not Executable(f_in_path):
        self.fail('Expected casting Executable(f_in_path) to bool=True')
    self._check_file(Executable(f_in_path).path(), os.path.join(pathdir, f_in_path))
    self._check_file('%s' % Executable(f_in_path), os.path.join(pathdir, f_in_path))
    self._check_file(Executable(f_in_path).executable, os.path.join(pathdir, f_in_path))
    self.assertFalse(Executable(f_in_tmp).available())
    if Executable(f_in_tmp):
        self.fail('Expected casting Executable(f_in_tmp) to bool=False')
    self.assertIsNone(Executable(f_in_tmp).path())
    self.assertEqual('%s' % Executable(f_in_tmp), '')
    self.assertIsNone(Executable(f_in_tmp).executable)
    Executable.pathlist = []
    self.assertFalse(Executable(f_in_cfg).available())
    Executable.pathlist.append(config_bindir)
    self.assertFalse(Executable(f_in_cfg).available())
    Executable.rehash()
    self.assertTrue(Executable(f_in_cfg).available())
    self.assertEqual(Executable(f_in_cfg).path(), os.path.join(config_bindir, f_in_cfg))
    Executable.pathlist = None
    Executable.rehash()
    self.assertTrue(Executable(f_in_cfg).available())
    self.assertEqual(Executable(f_in_cfg).path(), os.path.join(config_bindir, f_in_cfg))
    f_in_path2 = 'f_in_path2'
    f_loc = os.path.join(pathdir, f_in_path2)
    self.assertFalse(Executable(f_in_path2).available())
    output = StringIO()
    with LoggingIntercept(output, 'pyomo.common', logging.WARNING):
        Executable(f_in_path2).executable = f_loc
        self.assertIn("explicitly setting the path for '%s' to an invalid object or nonexistent location ('%s')" % (f_in_path2, f_loc), output.getvalue())
    self.assertFalse(Executable(f_in_path2).available())
    self._make_exec(os.path.join(pathdir, f_in_path2))
    self.assertFalse(Executable(f_in_path2).available())
    Executable(f_in_path2).rehash()
    self.assertTrue(Executable(f_in_path2).available())
    Executable(f_in_path2).disable()
    self.assertFalse(Executable(f_in_path2).available())
    self.assertIsNone(Executable(f_in_path2).path())
    Executable(f_in_path2).rehash()
    self.assertTrue(Executable(f_in_path2).available())
    self.assertEqual(Executable(f_in_path2).path(), f_loc)