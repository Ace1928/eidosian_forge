from Cython.Build.Cythonize import (
from Cython.Compiler import Options
from Cython.Compiler.Tests.Utils import backup_Options, restore_Options, check_global_options
from unittest import TestCase
import sys
def test_directive_value_no(self):
    options, args = self.parse_args(['-X', 'cdivision=no'])
    self.assertFalse(args)
    self.assertTrue(self.are_default(options, ['directives']))
    self.assertEqual(options.directives['cdivision'], False)