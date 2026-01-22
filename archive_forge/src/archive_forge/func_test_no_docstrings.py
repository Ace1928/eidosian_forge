from Cython.Build.Cythonize import (
from Cython.Compiler import Options
from Cython.Compiler.Tests.Utils import backup_Options, restore_Options, check_global_options
from unittest import TestCase
import sys
def test_no_docstrings(self):
    options, sources = parse_args(['foo.pyx', '--no-docstrings'])
    self.assertEqual(sources, ['foo.pyx'])
    self.assertEqual(Options.docstrings, False)
    self.check_default_global_options(['docstrings'])