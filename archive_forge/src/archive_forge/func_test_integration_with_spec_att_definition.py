import unittest
from unittest import mock
def test_integration_with_spec_att_definition(self):
    """You are not restricted when using mock with spec"""
    m = mock.Mock(SampleObject)
    m.attr_sample1 = 1
    m.attr_sample3 = 3
    mock.seal(m)
    self.assertEqual(m.attr_sample1, 1)
    self.assertEqual(m.attr_sample3, 3)
    with self.assertRaises(AttributeError):
        m.attr_sample2