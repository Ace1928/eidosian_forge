from Cython.Build.Cythonize import (
from Cython.Compiler import Options
from Cython.Compiler.Tests.Utils import backup_Options, restore_Options, check_global_options
from unittest import TestCase
import sys
def test_annotate_and_optional(self):
    options, args = self.parse_args(['-a', '--3str'])
    self.assertFalse(args)
    self.assertTrue(self.are_default(options, ['annotate', 'language_level']))
    self.assertEqual(options.annotate, 'default')
    self.assertEqual(options.language_level, '3str')