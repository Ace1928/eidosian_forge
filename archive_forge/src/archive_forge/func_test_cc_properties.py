import contextlib
import importlib
import os
import shutil
import subprocess
import sys
import tempfile
from unittest import skip
from ctypes import *
import numpy as np
import llvmlite.binding as ll
from numba.core import utils
from numba.tests.support import (TestCase, tag, import_dynamic, temp_directory,
import unittest
def test_cc_properties(self):
    cc = self._test_module.cc
    self.assertEqual(cc.name, 'pycc_test_simple')
    d = self._test_module.cc.output_dir
    self.assertTrue(os.path.isdir(d), d)
    f = self._test_module.cc.output_file
    self.assertFalse(os.path.exists(f), f)
    self.assertTrue(os.path.basename(f).startswith('pycc_test_simple.'), f)
    if sys.platform.startswith('linux'):
        self.assertTrue(f.endswith('.so'), f)
        from numba.pycc.platform import find_pyext_ending
        self.assertIn(find_pyext_ending(), f)