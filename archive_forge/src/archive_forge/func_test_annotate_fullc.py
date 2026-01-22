from Cython.Build.Cythonize import (
from Cython.Compiler import Options
from Cython.Compiler.Tests.Utils import backup_Options, restore_Options, check_global_options
from unittest import TestCase
import sys
def test_annotate_fullc(self):
    options, sources = parse_args(['foo.pyx', '--annotate-fullc'])
    self.assertEqual(sources, ['foo.pyx'])
    self.assertEqual(Options.annotate, 'fullc')
    self.check_default_global_options(['annotate'])