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
def test_spec_list_subclass(self):

    class Sub(list):
        pass
    mock = Mock(spec=Sub(['foo']))
    mock.append(3)
    mock.append.assert_called_with(3)
    self.assertRaises(AttributeError, getattr, mock, 'foo')