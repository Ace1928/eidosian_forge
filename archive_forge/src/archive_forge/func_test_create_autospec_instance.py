import unittest2 as unittest
from mock.tests.support import is_instance, X, SomeClass
from mock import (
def test_create_autospec_instance(self):
    mock = create_autospec(SomeClass, instance=True)
    self.assertRaises(TypeError, mock)
    mock.wibble()
    mock.wibble.assert_called_once_with()
    self.assertRaises(TypeError, mock.wibble, 'some', 'args')