import os
import sys
import unittest
from distutils.command.build_py import build_py
from distutils.core import Distribution
from distutils.errors import DistutilsFileError
from distutils.tests import support
from test.support import requires_subprocess
@unittest.skipIf(sys.dont_write_bytecode, 'byte-compile disabled')
@requires_subprocess()
def test_byte_compile_optimized(self):
    project_dir, dist = self.create_dist(py_modules=['boiledeggs'])
    os.chdir(project_dir)
    self.write_file('boiledeggs.py', 'import antigravity')
    cmd = build_py(dist)
    cmd.compile = 0
    cmd.optimize = 1
    cmd.build_lib = 'here'
    cmd.finalize_options()
    cmd.run()
    found = os.listdir(cmd.build_lib)
    self.assertEqual(sorted(found), ['__pycache__', 'boiledeggs.py'])
    found = os.listdir(os.path.join(cmd.build_lib, '__pycache__'))
    expect = 'boiledeggs.{}.opt-1.pyc'.format(sys.implementation.cache_tag)
    self.assertEqual(sorted(found), [expect])