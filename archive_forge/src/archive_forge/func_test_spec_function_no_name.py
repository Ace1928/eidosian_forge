import inspect
import time
import types
import unittest
from unittest.mock import (
from datetime import datetime
from functools import partial
def test_spec_function_no_name(self):
    func = lambda: 'nope'
    mock = create_autospec(func)
    self.assertEqual(mock.__name__, 'funcopy')