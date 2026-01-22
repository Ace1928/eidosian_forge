import unittest
from mock import Mock, call
from .watch import Watch
def test_unmarshal_with_no_return_type(self):
    w = Watch()
    event = w.unmarshal_event('{"type": "ADDED", "object": ["test1"]}', None)
    self.assertEqual('ADDED', event['type'])
    self.assertEqual(['test1'], event['object'])
    self.assertEqual(['test1'], event['raw_object'])