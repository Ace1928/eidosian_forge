import copy
import pickle
import sys
import tempfile
import six
import unittest2 as unittest
import mock
from mock import (
from mock.mock import _CallList
from mock.tests.support import (
def test_side_effect_iterator_exceptions(self):
    for Klass in (Mock, MagicMock):
        iterable = (ValueError, 3, KeyError, 6)
        m = Klass(side_effect=iterable)
        self.assertRaises(ValueError, m)
        self.assertEqual(m(), 3)
        self.assertRaises(KeyError, m)
        self.assertEqual(m(), 6)