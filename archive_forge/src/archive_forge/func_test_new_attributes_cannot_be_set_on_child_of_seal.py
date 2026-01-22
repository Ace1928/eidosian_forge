import unittest
from unittest import mock
def test_new_attributes_cannot_be_set_on_child_of_seal(self):
    m = mock.Mock()
    m.test.test2 = 1
    mock.seal(m)
    with self.assertRaises(AttributeError):
        m.test.test3 = 1