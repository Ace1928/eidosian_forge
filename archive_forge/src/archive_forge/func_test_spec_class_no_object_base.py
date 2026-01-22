import copy
import re
import sys
import tempfile
from test.support import ALWAYS_EQ
import unittest
from unittest.test.testmock.support import is_instance
from unittest import mock
from unittest.mock import (
def test_spec_class_no_object_base(self):

    class X:
        pass
    mock = Mock(spec=X)
    self.assertIsInstance(mock, X)
    mock = Mock(spec=X())
    self.assertIsInstance(mock, X)
    self.assertIs(mock.__class__, X)
    self.assertEqual(Mock().__class__.__name__, 'Mock')
    mock = Mock(spec_set=X)
    self.assertIsInstance(mock, X)
    mock = Mock(spec_set=X())
    self.assertIsInstance(mock, X)