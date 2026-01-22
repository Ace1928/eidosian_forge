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
def test_sorted_call_signature(self):
    m = Mock()
    m.hello(name='hello', daddy='hero')
    text = "call(daddy='hero', name='hello')"
    self.assertEqual(repr(m.hello.call_args), text)