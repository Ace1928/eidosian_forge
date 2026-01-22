import inspect
import time
import types
import unittest
from unittest.mock import (
from datetime import datetime
from functools import partial
def test_call_with_name(self):
    self.assertEqual(_Call((), 'foo')[0], 'foo')
    self.assertEqual(_Call((('bar', 'barz'),))[0], '')
    self.assertEqual(_Call((('bar', 'barz'), {'hello': 'world'}))[0], '')