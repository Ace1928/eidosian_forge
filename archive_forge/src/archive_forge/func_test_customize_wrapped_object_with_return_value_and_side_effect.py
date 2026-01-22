import copy
import re
import sys
import tempfile
from test.support import ALWAYS_EQ
import unittest
from unittest.test.testmock.support import is_instance
from unittest import mock
from unittest.mock import (
def test_customize_wrapped_object_with_return_value_and_side_effect(self):

    class Real(object):

        def method(self):
            pass
    real = Real()
    mock = Mock(wraps=real)
    mock.method.side_effect = [sentinel.VALUE1, sentinel.VALUE2]
    mock.method.return_value = sentinel.WRONG_VALUE
    self.assertEqual(mock.method(), sentinel.VALUE1)
    self.assertEqual(mock.method(), sentinel.VALUE2)
    self.assertRaises(StopIteration, mock.method)