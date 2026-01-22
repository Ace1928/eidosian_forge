import unittest2 as unittest
from mock.tests.support import is_instance, X, SomeClass
from mock import (
def test_patch_spec_set(self):
    patcher = patch('%s.X' % __name__, spec_set=True)
    mock = patcher.start()
    self.addCleanup(patcher.stop)
    instance = mock()
    mock.assert_called_once_with()
    self.assertNotCallable(instance)
    self.assertRaises(TypeError, instance)