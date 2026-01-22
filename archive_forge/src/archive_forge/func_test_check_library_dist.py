import unittest
import os
import sys
import sysconfig
from test.support import (
from distutils.command.build_clib import build_clib
from distutils.errors import DistutilsSetupError
from distutils.tests import support
def test_check_library_dist(self):
    pkg_dir, dist = self.create_dist()
    cmd = build_clib(dist)
    self.assertRaises(DistutilsSetupError, cmd.check_library_list, 'foo')
    self.assertRaises(DistutilsSetupError, cmd.check_library_list, ['foo1', 'foo2'])
    self.assertRaises(DistutilsSetupError, cmd.check_library_list, [(1, 'foo1'), ('name', 'foo2')])
    self.assertRaises(DistutilsSetupError, cmd.check_library_list, [('name', 'foo1'), ('another/name', 'foo2')])
    self.assertRaises(DistutilsSetupError, cmd.check_library_list, [('name', {}), ('another', 'foo2')])
    libs = [('name', {}), ('name', {'ok': 'good'})]
    cmd.check_library_list(libs)