from __future__ import division
import inspect
import sys
import textwrap
import six
import unittest2 as unittest
from mock import Mock, MagicMock
from mock.mock import _magics
def test_iterable_as_iter_return_value(self):
    m = MagicMock()
    m.__iter__.return_value = [1, 2, 3]
    self.assertEqual(list(m), [1, 2, 3])
    self.assertEqual(list(m), [1, 2, 3])
    m.__iter__.return_value = iter([4, 5, 6])
    self.assertEqual(list(m), [4, 5, 6])
    self.assertEqual(list(m), [])