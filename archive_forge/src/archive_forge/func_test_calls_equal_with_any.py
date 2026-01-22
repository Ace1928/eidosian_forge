import copy
import re
import sys
import tempfile
from test.support import ALWAYS_EQ
import unittest
from unittest.test.testmock.support import is_instance
from unittest import mock
from unittest.mock import (
def test_calls_equal_with_any(self):
    mm = mock.MagicMock()
    self.assertTrue(mm == mm)
    self.assertFalse(mm != mm)
    self.assertFalse(mm == mock.MagicMock())
    self.assertTrue(mm != mock.MagicMock())
    self.assertTrue(mm == mock.ANY)
    self.assertFalse(mm != mock.ANY)
    self.assertTrue(mock.ANY == mm)
    self.assertFalse(mock.ANY != mm)
    self.assertTrue(mm == ALWAYS_EQ)
    self.assertFalse(mm != ALWAYS_EQ)
    call1 = mock.call(mock.MagicMock())
    call2 = mock.call(mock.ANY)
    self.assertTrue(call1 == call2)
    self.assertFalse(call1 != call2)
    self.assertTrue(call2 == call1)
    self.assertFalse(call2 != call1)
    self.assertTrue(call1 == ALWAYS_EQ)
    self.assertFalse(call1 != ALWAYS_EQ)
    self.assertFalse(call1 == 1)
    self.assertTrue(call1 != 1)