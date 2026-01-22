from Cython.Build.Cythonize import (
from Cython.Compiler import Options
from Cython.Compiler.Tests.Utils import backup_Options, restore_Options, check_global_options
from unittest import TestCase
import sys
def test_file_inbetween(self):
    options, args = self.parse_args(['-i', 'file.pyx', '-a'])
    self.assertEqual(args, ['file.pyx'])
    self.assertEqual(options.build_inplace, True)
    self.assertEqual(options.annotate, 'default')
    self.assertTrue(self.are_default(options, ['build_inplace', 'annotate']))