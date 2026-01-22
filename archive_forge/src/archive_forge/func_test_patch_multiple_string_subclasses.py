import os
import sys
import six
import unittest2 as unittest
from mock.tests import support
from mock.tests.support import SomeClass, is_instance, callable
from mock import (
from mock.mock import _patch, _get_target
def test_patch_multiple_string_subclasses(self):
    for base in (str, unicode):
        Foo = type('Foo', (base,), {'fish': 'tasty'})
        foo = Foo()

        @patch.multiple(foo, fish='nearly gone')
        def test():
            self.assertEqual(foo.fish, 'nearly gone')
        test()
        self.assertEqual(foo.fish, 'tasty')