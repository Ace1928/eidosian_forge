import os
import sys
import six
import unittest2 as unittest
from mock.tests import support
from mock.tests.support import SomeClass, is_instance, callable
from mock import (
from mock.mock import _patch, _get_target
def test_single_patchobject(self):

    class Something(object):
        attribute = sentinel.Original

    @patch.object(Something, 'attribute', sentinel.Patched)
    def test():
        self.assertEqual(Something.attribute, sentinel.Patched, 'unpatched')
    test()
    self.assertEqual(Something.attribute, sentinel.Original, 'patch not restored')