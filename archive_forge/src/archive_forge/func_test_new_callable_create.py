import os
import sys
import six
import unittest2 as unittest
from mock.tests import support
from mock.tests.support import SomeClass, is_instance, callable
from mock import (
from mock.mock import _patch, _get_target
def test_new_callable_create(self):
    non_existent_attr = '%s.weeeee' % foo_name
    p = patch(non_existent_attr, new_callable=NonCallableMock)
    self.assertRaises(AttributeError, p.start)
    p = patch(non_existent_attr, new_callable=NonCallableMock, create=True)
    m = p.start()
    try:
        self.assertNotCallable(m, magic=False)
    finally:
        p.stop()