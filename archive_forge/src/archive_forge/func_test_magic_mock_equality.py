from __future__ import division
import inspect
import sys
import textwrap
import six
import unittest2 as unittest
from mock import Mock, MagicMock
from mock.mock import _magics
def test_magic_mock_equality(self):
    mock = MagicMock()
    self.assertIsInstance(mock == object(), bool)
    self.assertIsInstance(mock != object(), bool)
    self.assertEqual(mock == object(), False)
    self.assertEqual(mock != object(), True)
    self.assertEqual(mock == mock, True)
    self.assertEqual(mock != mock, False)