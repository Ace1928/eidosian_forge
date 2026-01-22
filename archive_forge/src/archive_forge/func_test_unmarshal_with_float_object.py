import unittest
from mock import Mock, call
from .watch import Watch
def test_unmarshal_with_float_object(self):
    w = Watch()
    event = w.unmarshal_event('{"type": "ADDED", "object": 1}', 'float')
    self.assertEqual('ADDED', event['type'])
    self.assertEqual(1.0, event['object'])
    self.assertTrue(isinstance(event['object'], float))
    self.assertEqual(1, event['raw_object'])