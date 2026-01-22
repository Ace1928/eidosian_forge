import ctypes
import threading
from ctypes import CFUNCTYPE, c_int, c_int32
from ctypes.util import find_library
import gc
import locale
import os
import platform
import re
import subprocess
import sys
import unittest
from contextlib import contextmanager
from tempfile import mkstemp
from llvmlite import ir
from llvmlite import binding as llvm
from llvmlite.binding import ffi
from llvmlite.tests import TestCase
def test_function_arguments(self):
    mod = self.module()
    func = mod.get_function('sum')
    self.assertTrue(func.is_function)
    args = list(func.arguments)
    self.assertEqual(len(args), 2)
    self.assertTrue(args[0].is_argument)
    self.assertTrue(args[1].is_argument)
    self.assertEqual(args[0].name, '.1')
    self.assertEqual(str(args[0].type), 'i32')
    self.assertEqual(args[1].name, '.2')
    self.assertEqual(str(args[1].type), 'i32')
    with self.assertRaises(ValueError):
        args[0].blocks
    with self.assertRaises(ValueError):
        args[0].arguments