import copy
import re
import sys
import tempfile
from test.support import ALWAYS_EQ
import unittest
from unittest.test.testmock.support import is_instance
from unittest import mock
from unittest.mock import (
def test_customize_wrapped_object_with_side_effect_iterable_with_default(self):

    class Real(object):

        def method(self):
            return sentinel.ORIGINAL_VALUE
    real = Real()
    mock = Mock(wraps=real)
    mock.method.side_effect = [sentinel.VALUE1, DEFAULT]
    self.assertEqual(mock.method(), sentinel.VALUE1)
    self.assertEqual(mock.method(), sentinel.ORIGINAL_VALUE)
    self.assertRaises(StopIteration, mock.method)