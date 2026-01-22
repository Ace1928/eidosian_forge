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
def test_attribute_deletion(self):
    for mock in (Mock(), MagicMock(), NonCallableMagicMock(), NonCallableMock()):
        self.assertTrue(hasattr(mock, 'm'))
        del mock.m
        self.assertFalse(hasattr(mock, 'm'))
        del mock.f
        self.assertFalse(hasattr(mock, 'f'))
        self.assertRaises(AttributeError, getattr, mock, 'f')