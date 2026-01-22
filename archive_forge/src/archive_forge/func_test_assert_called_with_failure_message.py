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
def test_assert_called_with_failure_message(self):
    mock = NonCallableMock()
    expected = "mock(1, '2', 3, bar='foo')"
    message = 'Expected call: %s\nNot called'
    self.assertRaisesWithMsg(AssertionError, message % (expected,), mock.assert_called_with, 1, '2', 3, bar='foo')
    mock.foo(1, '2', 3, foo='foo')
    asserters = [mock.foo.assert_called_with, mock.foo.assert_called_once_with]
    for meth in asserters:
        actual = "foo(1, '2', 3, foo='foo')"
        expected = "foo(1, '2', 3, bar='foo')"
        message = 'Expected call: %s\nActual call: %s'
        self.assertRaisesWithMsg(AssertionError, message % (expected, actual), meth, 1, '2', 3, bar='foo')
    for meth in asserters:
        actual = "foo(1, '2', 3, foo='foo')"
        expected = "foo(bar='foo')"
        message = 'Expected call: %s\nActual call: %s'
        self.assertRaisesWithMsg(AssertionError, message % (expected, actual), meth, bar='foo')
    for meth in asserters:
        actual = "foo(1, '2', 3, foo='foo')"
        expected = 'foo(1, 2, 3)'
        message = 'Expected call: %s\nActual call: %s'
        self.assertRaisesWithMsg(AssertionError, message % (expected, actual), meth, 1, 2, 3)
    for meth in asserters:
        actual = "foo(1, '2', 3, foo='foo')"
        expected = 'foo()'
        message = 'Expected call: %s\nActual call: %s'
        self.assertRaisesWithMsg(AssertionError, message % (expected, actual), meth)