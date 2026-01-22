import os
import sys
import six
import unittest2 as unittest
from mock.tests import support
from mock.tests.support import SomeClass, is_instance, callable
from mock import (
from mock.mock import _patch, _get_target
def test_new_callable_inherit_non_mock(self):

    class NotAMock(object):

        def __init__(self, spec):
            self.spec = spec
    p = patch(foo_name, new_callable=NotAMock, spec=True)
    m = p.start()
    try:
        self.assertTrue(is_instance(m, NotAMock))
        self.assertRaises(AttributeError, getattr, m, 'return_value')
    finally:
        p.stop()
    self.assertEqual(m.spec, Foo)