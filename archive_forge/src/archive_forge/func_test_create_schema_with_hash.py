from tests.unit import unittest
from mock import Mock
from boto.dynamodb.layer2 import Layer2
from boto.dynamodb.table import Table, Schema
def test_create_schema_with_hash(self):
    schema = self.layer2.create_schema('foo', str)
    self.assertEqual(schema.hash_key_name, 'foo')
    self.assertEqual(schema.hash_key_type, 'S')
    self.assertIsNone(schema.range_key_name)
    self.assertIsNone(schema.range_key_type)