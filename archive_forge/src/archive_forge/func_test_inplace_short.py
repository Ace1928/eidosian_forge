from Cython.Build.Cythonize import (
from Cython.Compiler import Options
from Cython.Compiler.Tests.Utils import backup_Options, restore_Options, check_global_options
from unittest import TestCase
import sys
def test_inplace_short(self):
    options, args = self.parse_args(['-i'])
    self.assertFalse(args)
    self.assertTrue(self.are_default(options, ['build_inplace']))
    self.assertEqual(options.build_inplace, True)