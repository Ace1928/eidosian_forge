import copy
import re
import sys
import tempfile
from test.support import ALWAYS_EQ
import unittest
from unittest.test.testmock.support import is_instance
from unittest import mock
from unittest.mock import (
def test_assert_called_exception_message(self):
    msg = "Expected '{0}' to have been called"
    with self.assertRaisesRegex(AssertionError, msg.format('mock')):
        Mock().assert_called()
    with self.assertRaisesRegex(AssertionError, msg.format('test_name')):
        Mock(name='test_name').assert_called()