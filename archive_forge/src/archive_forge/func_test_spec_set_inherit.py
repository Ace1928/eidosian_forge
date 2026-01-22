import os
import sys
import six
import unittest2 as unittest
from mock.tests import support
from mock.tests.support import SomeClass, is_instance, callable
from mock import (
from mock.mock import _patch, _get_target
def test_spec_set_inherit(self):

    @patch('%s.SomeClass' % __name__, spec_set=True)
    def test(MockClass):
        instance = MockClass()
        instance.z = 'foo'
    self.assertRaises(AttributeError, test)