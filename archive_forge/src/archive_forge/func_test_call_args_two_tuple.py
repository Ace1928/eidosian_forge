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
def test_call_args_two_tuple(self):
    mock = Mock()
    mock(1, a=3)
    mock(2, b=4)
    self.assertEqual(len(mock.call_args), 2)
    args, kwargs = mock.call_args
    self.assertEqual(args, (2,))
    self.assertEqual(kwargs, dict(b=4))
    expected_list = [((1,), dict(a=3)), ((2,), dict(b=4))]
    for expected, call_args in zip(expected_list, mock.call_args_list):
        self.assertEqual(len(call_args), 2)
        self.assertEqual(expected[0], call_args[0])
        self.assertEqual(expected[1], call_args[1])