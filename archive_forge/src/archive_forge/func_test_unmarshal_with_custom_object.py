import unittest
from mock import Mock, call
from .watch import Watch
def test_unmarshal_with_custom_object(self):
    w = Watch()
    event = w.unmarshal_event('{"type": "ADDED", "object": {"apiVersion":"test.com/v1beta1","kind":"foo","metadata":{"name": "bar", "resourceVersion": "1"}}}', 'object')
    self.assertEqual('ADDED', event['type'])
    self.assertTrue(isinstance(event['object'], dict))
    self.assertEqual('1', event['object']['metadata']['resourceVersion'])
    self.assertEqual('1', w.resource_version)