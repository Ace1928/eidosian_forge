import operator
import sys
import types
import unittest
import abc
import pytest
import six
def test_assertNotRegex():

    class TestAssertNotRegex(unittest.TestCase):

        def test(self):
            with self.assertRaises(AssertionError):
                six.assertNotRegex(self, 'test', '^t')
            six.assertNotRegex(self, 'test', '^a')
    TestAssertNotRegex('test').test()