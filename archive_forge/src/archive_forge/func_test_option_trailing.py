from Cython.Build.Cythonize import (
from Cython.Compiler import Options
from Cython.Compiler.Tests.Utils import backup_Options, restore_Options, check_global_options
from unittest import TestCase
import sys
def test_option_trailing(self):
    options, args = self.parse_args(['file.pyx', '-i'])
    self.assertEqual(args, ['file.pyx'])
    self.assertEqual(options.build_inplace, True)
    self.assertTrue(self.are_default(options, ['build_inplace']))