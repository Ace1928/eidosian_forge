from Cython.Build.Cythonize import (
from Cython.Compiler import Options
from Cython.Compiler.Tests.Utils import backup_Options, restore_Options, check_global_options
from unittest import TestCase
import sys
def test_lenient(self):
    options, sources = parse_args(['foo.pyx', '--lenient'])
    self.assertEqual(sources, ['foo.pyx'])
    self.assertEqual(Options.error_on_unknown_names, False)
    self.assertEqual(Options.error_on_uninitialized, False)
    self.check_default_global_options(['error_on_unknown_names', 'error_on_uninitialized'])