import os
import sys
import six
import unittest2 as unittest
from mock.tests import support
from mock.tests.support import SomeClass, is_instance, callable
from mock import (
from mock.mock import _patch, _get_target
def test_patch_keyword_args(self):
    kwargs = {'side_effect': KeyError, 'foo.bar.return_value': 33, 'foo': MagicMock()}
    patcher = patch(foo_name, **kwargs)
    mock = patcher.start()
    patcher.stop()
    self.assertRaises(KeyError, mock)
    self.assertEqual(mock.foo.bar(), 33)
    self.assertIsInstance(mock.foo, MagicMock)