import copy
import re
import sys
import tempfile
from test.support import ALWAYS_EQ
import unittest
from unittest.test.testmock.support import is_instance
from unittest import mock
from unittest.mock import (
def test_parent_propagation_with_autospec_attach_mock(self):

    def foo(a, b):
        pass
    parent = Mock()
    parent.attach_mock(create_autospec(foo, name='bar'), 'child')
    parent.child(1, 2)
    self.assertRaises(TypeError, parent.child, 1)
    self.assertEqual(parent.child.mock_calls, [call.child(1, 2)])
    self.assertIn('mock.child', repr(parent.child.mock))