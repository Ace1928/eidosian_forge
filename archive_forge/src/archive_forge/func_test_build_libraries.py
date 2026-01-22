import unittest
import os
import sys
import sysconfig
from test.support import (
from distutils.command.build_clib import build_clib
from distutils.errors import DistutilsSetupError
from distutils.tests import support
def test_build_libraries(self):
    pkg_dir, dist = self.create_dist()
    cmd = build_clib(dist)

    class FakeCompiler:

        def compile(*args, **kw):
            pass
        create_static_lib = compile
    cmd.compiler = FakeCompiler()
    lib = [('name', {'sources': 'notvalid'})]
    self.assertRaises(DistutilsSetupError, cmd.build_libraries, lib)
    lib = [('name', {'sources': list()})]
    cmd.build_libraries(lib)
    lib = [('name', {'sources': tuple()})]
    cmd.build_libraries(lib)