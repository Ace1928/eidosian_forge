import os
import sys
import inspect
import contextlib
import numpy as np
import logging
from io import StringIO
import unittest
from numba.tests.support import SerialMixin, create_temp_module
from numba.core import dispatcher
from numba import jit_module
import numpy as np
from numba import jit, jit_module
def test_create_temp_jitted_module(self):
    sys_path_original = list(sys.path)
    sys_modules_original = dict(sys.modules)
    with create_temp_module(self.source_lines) as test_module:
        temp_module_dir = os.path.dirname(test_module.__file__)
        self.assertEqual(temp_module_dir, sys.path[0])
        self.assertEqual(sys.path[1:], sys_path_original)
        self.assertTrue(test_module.__name__ in sys.modules)
    self.assertEqual(sys.path, sys_path_original)
    self.assertEqual(sys.modules, sys_modules_original)