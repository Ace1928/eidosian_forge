from __future__ import print_function
from __future__ import absolute_import
import subprocess
import unittest
import greenlet
from . import _test_extension_cpp
from . import TestCase
from . import WIN
def test_unhandled_std_exception_aborts(self):
    self._do_test_unhandled_exception(_test_extension_cpp.test_exception_throw_std)