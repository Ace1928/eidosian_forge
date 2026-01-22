from Cython.Build.Cythonize import (
from Cython.Compiler import Options
from Cython.Compiler.Tests.Utils import backup_Options, restore_Options, check_global_options
from unittest import TestCase
import sys
def test_lenient_long(self):
    options, args = self.parse_args(['--lenient'])
    self.assertTrue(self.are_default(options, ['lenient']))
    self.assertFalse(args)
    self.assertEqual(options.lenient, True)