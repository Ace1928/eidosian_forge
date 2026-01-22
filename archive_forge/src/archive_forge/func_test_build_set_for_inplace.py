from Cython.Build.Cythonize import (
from Cython.Compiler import Options
from Cython.Compiler.Tests.Utils import backup_Options, restore_Options, check_global_options
from unittest import TestCase
import sys
def test_build_set_for_inplace(self):
    options, args = parse_args(['foo.pyx', '-i'])
    self.assertEqual(options.build, True)
    self.check_default_global_options()