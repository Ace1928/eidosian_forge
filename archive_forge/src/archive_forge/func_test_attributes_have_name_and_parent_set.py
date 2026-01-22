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
def test_attributes_have_name_and_parent_set(self):
    mock = Mock()
    something = mock.something
    self.assertEqual(something._mock_name, 'something', 'attribute name not set correctly')
    self.assertEqual(something._mock_parent, mock, 'attribute parent not set correctly')