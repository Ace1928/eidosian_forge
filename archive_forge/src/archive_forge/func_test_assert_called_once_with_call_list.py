import copy
import re
import sys
import tempfile
from test.support import ALWAYS_EQ
import unittest
from unittest.test.testmock.support import is_instance
from unittest import mock
from unittest.mock import (
def test_assert_called_once_with_call_list(self):
    m = Mock()
    m(1)
    m(2)
    self.assertRaisesRegex(AssertionError, re.escape('Calls: [call(1), call(2)]'), lambda: m.assert_called_once_with(2))