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
def test_adding_return_value_mock(self):
    for Klass in (Mock, MagicMock):
        mock = Klass()
        mock.return_value = MagicMock()
        mock()()
        self.assertEqual(mock.mock_calls, [call(), call()()])