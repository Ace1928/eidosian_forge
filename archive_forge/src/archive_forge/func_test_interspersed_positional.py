from Cython.Build.Cythonize import (
from Cython.Compiler import Options
from Cython.Compiler.Tests.Utils import backup_Options, restore_Options, check_global_options
from unittest import TestCase
import sys
def test_interspersed_positional(self):
    options, sources = self.parse_args(['file1.pyx', '-a', 'file2.pyx'])
    self.assertEqual(sources, ['file1.pyx', 'file2.pyx'])
    self.assertEqual(options.annotate, 'default')
    self.assertTrue(self.are_default(options, ['annotate']))