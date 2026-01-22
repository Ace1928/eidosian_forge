import os
import sys
import six
import unittest2 as unittest
from mock.tests import support
from mock.tests.support import SomeClass, is_instance, callable
from mock import (
from mock.mock import _patch, _get_target
def test_patchobject_with_default_mock(self):

    class Test(object):
        something = sentinel.Original
        something2 = sentinel.Original2

    @patch.object(Test, 'something')
    def test(mock):
        self.assertEqual(mock, Test.something, 'Mock not passed into test function')
        self.assertIsInstance(mock, MagicMock, 'patch with two arguments did not create a mock')
    test()

    @patch.object(Test, 'something')
    @patch.object(Test, 'something2')
    def test(this1, this2, mock1, mock2):
        self.assertEqual(this1, sentinel.this1, "Patched function didn't receive initial argument")
        self.assertEqual(this2, sentinel.this2, "Patched function didn't receive second argument")
        self.assertEqual(mock1, Test.something2, 'Mock not passed into test function')
        self.assertEqual(mock2, Test.something, 'Second Mock not passed into test function')
        self.assertIsInstance(mock2, MagicMock, 'patch with two arguments did not create a mock')
        self.assertIsInstance(mock2, MagicMock, 'patch with two arguments did not create a mock')
        self.assertNotEqual(outerMock1, mock1, 'unexpected value for mock1')
        self.assertNotEqual(outerMock2, mock2, 'unexpected value for mock1')
        return (mock1, mock2)
    outerMock1 = outerMock2 = None
    outerMock1, outerMock2 = test(sentinel.this1, sentinel.this2)
    test(sentinel.this1, sentinel.this2)