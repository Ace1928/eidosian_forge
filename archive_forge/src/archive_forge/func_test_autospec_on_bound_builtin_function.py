import inspect
import time
import types
import unittest
from unittest.mock import (
from datetime import datetime
from functools import partial
def test_autospec_on_bound_builtin_function(self):
    meth = types.MethodType(time.ctime, time.time())
    self.assertIsInstance(meth(), str)
    mocked = create_autospec(meth)
    mocked()
    mocked.assert_called_once_with()
    mocked.reset_mock()
    mocked(4, 5, 6)
    mocked.assert_called_once_with(4, 5, 6)