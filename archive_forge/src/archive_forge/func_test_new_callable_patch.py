import os
import sys
import six
import unittest2 as unittest
from mock.tests import support
from mock.tests.support import SomeClass, is_instance, callable
from mock import (
from mock.mock import _patch, _get_target
def test_new_callable_patch(self):
    patcher = patch(foo_name, new_callable=NonCallableMagicMock)
    m1 = patcher.start()
    patcher.stop()
    m2 = patcher.start()
    patcher.stop()
    self.assertIsNot(m1, m2)
    for mock in (m1, m2):
        self.assertNotCallable(m1)