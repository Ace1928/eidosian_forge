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
@unittest.skipIf(six.PY3, 'no old style classes in Python 3')
def test_spec_set_old_style_classes(self):

    class Foo:
        bar = 7
    mock = Mock(spec_set=Foo)
    mock.bar = 6
    self.assertRaises(AttributeError, lambda: mock.foo)

    def _set():
        mock.foo = 3
    self.assertRaises(AttributeError, _set)
    mock = Mock(spec_set=Foo())
    mock.bar = 6
    self.assertRaises(AttributeError, lambda: mock.foo)

    def _set():
        mock.foo = 3
    self.assertRaises(AttributeError, _set)