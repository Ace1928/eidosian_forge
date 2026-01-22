from Cython.Build.Cythonize import (
from Cython.Compiler import Options
from Cython.Compiler.Tests.Utils import backup_Options, restore_Options, check_global_options
from unittest import TestCase
import sys
def test_interspersed_positional3(self):
    options, sources = self.parse_args(['-f', 'f1', 'f2', '-a', 'f3', 'f4', '-a', 'f5'])
    self.assertEqual(sources, ['f1', 'f2', 'f3', 'f4', 'f5'])
    self.assertEqual(options.annotate, 'default')
    self.assertEqual(options.force, True)
    self.assertTrue(self.are_default(options, ['annotate', 'force']))