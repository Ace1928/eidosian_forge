import unittest
import os
import sys
import sysconfig
from test.support import (
from distutils.command.config import dump_file, config
from distutils.tests import support
from distutils import log
@unittest.skipIf(sys.platform == 'win32', "can't test on Windows")
@requires_subprocess()
def test_search_cpp(self):
    cmd = missing_compiler_executable(['preprocessor'])
    if cmd is not None:
        self.skipTest('The %r command is not found' % cmd)
    pkg_dir, dist = self.create_dist()
    cmd = config(dist)
    cmd._check_compiler()
    compiler = cmd.compiler
    if sys.platform[:3] == 'aix' and 'xlc' in compiler.preprocessor[0].lower():
        self.skipTest('xlc: The -E option overrides the -P, -o, and -qsyntaxonly options')
    match = cmd.search_cpp(pattern='xxx', body='/* xxx */')
    self.assertEqual(match, 0)
    match = cmd.search_cpp(pattern='_configtest', body='/* xxx */')
    self.assertEqual(match, 1)