import copy
import re
import sys
import tempfile
from test.support import ALWAYS_EQ
import unittest
from unittest.test.testmock.support import is_instance
from unittest import mock
from unittest.mock import (
def test_parent_propagation_with_create_autospec(self):

    def foo(a, b):
        pass
    mock = Mock()
    mock.child = create_autospec(foo)
    mock.child(1, 2)
    self.assertRaises(TypeError, mock.child, 1)
    self.assertEqual(mock.mock_calls, [call.child(1, 2)])
    self.assertIn('mock.child', repr(mock.child.mock))