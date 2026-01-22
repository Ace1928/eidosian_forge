import os
import sys
import six
import unittest2 as unittest
from mock.tests import support
from mock.tests.support import SomeClass, is_instance, callable
from mock import (
from mock.mock import _patch, _get_target
def test_patch_class_attribute(self):

    @patch('%s.SomeClass.class_attribute' % __name__, sentinel.ClassAttribute)
    def test():
        self.assertEqual(PTModule.SomeClass.class_attribute, sentinel.ClassAttribute, 'unpatched')
    test()
    self.assertIsNone(PTModule.SomeClass.class_attribute, 'patch not restored')