from Cython.Build.Cythonize import (
from Cython.Compiler import Options
from Cython.Compiler.Tests.Utils import backup_Options, restore_Options, check_global_options
from unittest import TestCase
import sys
def check_default_global_options(self, white_list=[]):
    self.assertEqual(check_global_options(self._options_backup, white_list), '')