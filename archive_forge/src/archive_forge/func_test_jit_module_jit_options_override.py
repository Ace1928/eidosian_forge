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
def test_jit_module_jit_options_override(self):
    source_lines = '\nfrom numba import jit, jit_module\n\n@jit(nogil=True, forceobj=True)\ndef inc(x):\n    return x + 1\n\ndef add(x, y):\n    return x + y\n\njit_module({jit_options})\n'
    jit_options = {'nopython': True, 'error_model': 'numpy', 'boundscheck': False}
    with create_temp_module(source_lines=source_lines, **jit_options) as test_module:
        self.assertEqual(test_module.add.targetoptions, jit_options)
        self.assertEqual(test_module.inc.targetoptions, {'nogil': True, 'forceobj': True, 'boundscheck': None, 'nopython': False})