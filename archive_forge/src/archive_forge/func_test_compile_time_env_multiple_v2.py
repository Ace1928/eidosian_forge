from Cython.Build.Cythonize import (
from Cython.Compiler import Options
from Cython.Compiler.Tests.Utils import backup_Options, restore_Options, check_global_options
from unittest import TestCase
import sys
def test_compile_time_env_multiple_v2(self):
    options, args = self.parse_args(['-E', 'MYSIZE=10,ARRSIZE=11'])
    self.assertFalse(args)
    self.assertTrue(self.are_default(options, ['compile_time_env']))
    self.assertEqual(options.compile_time_env['MYSIZE'], 10)
    self.assertEqual(options.compile_time_env['ARRSIZE'], 11)