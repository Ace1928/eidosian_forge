import copy
import re
import sys
import tempfile
from test.support import ALWAYS_EQ
import unittest
from unittest.test.testmock.support import is_instance
from unittest import mock
from unittest.mock import (
def test_assert_has_calls_not_matching_spec_error(self):

    def f(x=None):
        pass
    mock = Mock(spec=f)
    mock(1)
    with self.assertRaisesRegex(AssertionError, '^{}$'.format(re.escape('Calls not found.\nExpected: [call()]\n  Actual: [call(1)]'))) as cm:
        mock.assert_has_calls([call()])
    self.assertIsNone(cm.exception.__cause__)
    with self.assertRaisesRegex(AssertionError, '^{}$'.format(re.escape("Error processing expected calls.\nErrors: [None, TypeError('too many positional arguments')]\nExpected: [call(), call(1, 2)]\n  Actual: [call(1)]"))) as cm:
        mock.assert_has_calls([call(), call(1, 2)])
    self.assertIsInstance(cm.exception.__cause__, TypeError)