import inspect
import time
import types
import unittest
from unittest.mock import (
from datetime import datetime
from functools import partial
def test_spec_function_reset_mock(self):

    def f(a):
        pass
    rv = Mock()
    mock = create_autospec(f, return_value=rv)
    mock(1)(2)
    self.assertEqual(mock.mock_calls, [call(1)])
    self.assertEqual(rv.mock_calls, [call(2)])
    mock.reset_mock()
    self.assertEqual(mock.mock_calls, [])
    self.assertEqual(rv.mock_calls, [])