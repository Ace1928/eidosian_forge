from Cython.Build.Cythonize import (
from Cython.Compiler import Options
from Cython.Compiler.Tests.Utils import backup_Options, restore_Options, check_global_options
from unittest import TestCase
import sys
def test_exclude_short(self):
    options, args = self.parse_args(['-x', '*.pyx'])
    self.assertFalse(args)
    self.assertTrue(self.are_default(options, ['excludes']))
    self.assertTrue('*.pyx' in options.excludes)