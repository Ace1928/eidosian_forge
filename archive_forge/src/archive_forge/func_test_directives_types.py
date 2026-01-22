from Cython.Build.Cythonize import (
from Cython.Compiler import Options
from Cython.Compiler.Tests.Utils import backup_Options, restore_Options, check_global_options
from unittest import TestCase
import sys
def test_directives_types(self):
    directives = {'auto_pickle': True, 'c_string_type': 'bytearray', 'c_string_type': 'bytes', 'c_string_type': 'str', 'c_string_type': 'bytearray', 'c_string_type': 'unicode', 'c_string_encoding': 'ascii', 'language_level': 2, 'language_level': 3, 'language_level': '3str', 'set_initial_path': 'my_initial_path'}
    for key, value in directives.items():
        cmd = '{key}={value}'.format(key=key, value=str(value))
        options, args = self.parse_args(['-X', cmd])
        self.assertFalse(args)
        self.assertTrue(self.are_default(options, ['directives']), msg='Error for option: ' + cmd)
        self.assertEqual(options.directives[key], value, msg='Error for option: ' + cmd)