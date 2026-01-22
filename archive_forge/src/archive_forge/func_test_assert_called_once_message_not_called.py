import copy
import re
import sys
import tempfile
from test.support import ALWAYS_EQ
import unittest
from unittest.test.testmock.support import is_instance
from unittest import mock
from unittest.mock import (
def test_assert_called_once_message_not_called(self):
    m = Mock()
    with self.assertRaises(AssertionError) as e:
        m.assert_called_once()
    self.assertNotIn('Calls:', str(e.exception))