import unittest
from unittest import mock
def test_existing_attributes_allowed_after_seal(self):
    m = mock.Mock()
    m.test.return_value = 3
    mock.seal(m)
    self.assertEqual(m.test(), 3)