import time
import uuid
from decimal import Decimal
from tests.unit import unittest
from boto.dynamodb.exceptions import DynamoDBKeyNotFoundError
from boto.dynamodb.exceptions import DynamoDBConditionalCheckFailedError
from boto.dynamodb.layer2 import Layer2
from boto.dynamodb.types import get_dynamodb_type, Binary
from boto.dynamodb.condition import BEGINS_WITH, CONTAINS, GT
from boto.compat import six, long_type
def test_binary_attrs(self):
    c = self.dynamodb
    schema = c.create_schema(self.hash_key_name, self.hash_key_proto_value, self.range_key_name, self.range_key_proto_value)
    index = int(time.time())
    table_name = 'test-%d' % index
    read_units = 5
    write_units = 5
    table = self.create_table(table_name, schema, read_units, write_units)
    table.refresh(wait_for_active=True)
    item1_key = 'Amazon S3'
    item1_range = 'S3 Thread 1'
    item1_attrs = {'Message': 'S3 Thread 1 message text', 'LastPostedBy': 'User A', 'Views': 0, 'Replies': 0, 'Answered': 0, 'BinaryData': Binary(b'\x01\x02\x03\x04'), 'BinarySequence': set([Binary(b'\x01\x02'), Binary(b'\x03\x04')]), 'Tags': set(['largeobject', 'multipart upload']), 'LastPostDateTime': '12/9/2011 11:36:03 PM'}
    item1 = table.new_item(item1_key, item1_range, item1_attrs)
    item1.put()
    retrieved = table.get_item(item1_key, item1_range, consistent_read=True)
    self.assertEqual(retrieved['Message'], 'S3 Thread 1 message text')
    self.assertEqual(retrieved['Views'], 0)
    self.assertEqual(retrieved['Tags'], set(['largeobject', 'multipart upload']))
    self.assertEqual(retrieved['BinaryData'], Binary(b'\x01\x02\x03\x04'))
    self.assertEqual(retrieved['BinaryData'], b'\x01\x02\x03\x04')
    self.assertEqual(retrieved['BinarySequence'], set([Binary(b'\x01\x02'), Binary(b'\x03\x04')]))