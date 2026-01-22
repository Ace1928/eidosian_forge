import unittest
from unittest import mock
def test_seals_dont_recurse_on_manual_attributes(self):
    m = mock.Mock(name='root_mock')
    m.test1.test2 = mock.Mock(name='not_sealed')
    m.test1.test2.test3 = 4
    mock.seal(m)
    self.assertEqual(m.test1.test2.test3, 4)
    m.test1.test2.test4
    m.test1.test2.test4 = 1