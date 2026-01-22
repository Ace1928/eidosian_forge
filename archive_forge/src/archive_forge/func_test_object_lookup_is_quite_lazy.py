import os
import sys
import six
import unittest2 as unittest
from mock.tests import support
from mock.tests.support import SomeClass, is_instance, callable
from mock import (
from mock.mock import _patch, _get_target
def test_object_lookup_is_quite_lazy(self):
    global something
    original = something

    @patch('%s.something' % __name__, sentinel.Something2)
    def test():
        pass
    try:
        something = sentinel.replacement_value
        test()
        self.assertEqual(something, sentinel.replacement_value)
    finally:
        something = original