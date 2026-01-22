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
def check_compile_for_cpu(self, cpu_name):
    cc = self._test_module.cc
    cc.target_cpu = cpu_name
    with self.check_cc_compiled(cc) as lib:
        res = lib.multi(123, 321)
        self.assertPreciseEqual(res, 123 * 321)
        self.assertEqual(lib.multi.__module__, 'pycc_test_simple')