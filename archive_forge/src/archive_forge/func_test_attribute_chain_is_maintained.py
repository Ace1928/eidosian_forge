import unittest
from unittest import mock
def test_attribute_chain_is_maintained(self):
    m = mock.Mock(name='mock_name')
    m.test1.test2.test3.test4
    mock.seal(m)
    with self.assertRaises(AttributeError) as cm:
        m.test1.test2.test3.test4.boom
    self.assertIn('mock_name.test1.test2.test3.test4.boom', str(cm.exception))