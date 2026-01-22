import copy
import re
import sys
import tempfile
from test.support import ALWAYS_EQ
import unittest
from unittest.test.testmock.support import is_instance
from unittest import mock
from unittest.mock import (
def test_attach_mock_patch_autospec(self):
    parent = Mock()
    with mock.patch(f'{__name__}.something', autospec=True) as mock_func:
        self.assertEqual(mock_func.mock._extract_mock_name(), 'something')
        parent.attach_mock(mock_func, 'child')
        parent.child(1)
        something(2)
        mock_func(3)
        parent_calls = [call.child(1), call.child(2), call.child(3)]
        child_calls = [call(1), call(2), call(3)]
        self.assertEqual(parent.mock_calls, parent_calls)
        self.assertEqual(parent.child.mock_calls, child_calls)
        self.assertEqual(something.mock_calls, child_calls)
        self.assertEqual(mock_func.mock_calls, child_calls)
        self.assertIn('mock.child', repr(parent.child.mock))
        self.assertEqual(mock_func.mock._extract_mock_name(), 'mock.child')