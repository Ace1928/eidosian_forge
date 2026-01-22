import time
import base64
from tests.unit import unittest
from boto.dynamodb.exceptions import DynamoDBKeyNotFoundError
from boto.dynamodb.exceptions import DynamoDBConditionalCheckFailedError
from boto.dynamodb.exceptions import DynamoDBValidationError
from boto.dynamodb.layer1 import Layer1
def test_binary_attributes(self):
    c = self.dynamodb
    result = self.create_table(self.table_name, self.schema, self.provisioned_throughput)
    result = c.describe_table(self.table_name)
    while result['Table']['TableStatus'] != 'ACTIVE':
        time.sleep(5)
        result = c.describe_table(self.table_name)
    item1_key = 'Amazon DynamoDB'
    item1_range = 'DynamoDB Thread 1'
    item1_data = {self.hash_key_name: {self.hash_key_type: item1_key}, self.range_key_name: {self.range_key_type: item1_range}, 'Message': {'S': 'DynamoDB thread 1 message text'}, 'LastPostedBy': {'S': 'User A'}, 'Views': {'N': '0'}, 'Replies': {'N': '0'}, 'BinaryData': {'B': base64.b64encode(b'\x01\x02\x03\x04').decode('utf-8')}, 'Answered': {'N': '0'}, 'Tags': {'SS': ['index', 'primarykey', 'table']}, 'LastPostDateTime': {'S': '12/9/2011 11:36:03 PM'}}
    result = c.put_item(self.table_name, item1_data)
    key1 = {'HashKeyElement': {self.hash_key_type: item1_key}, 'RangeKeyElement': {self.range_key_type: item1_range}}
    result = c.get_item(self.table_name, key=key1, consistent_read=True)
    self.assertEqual(result['Item']['BinaryData'], {'B': base64.b64encode(b'\x01\x02\x03\x04').decode('utf-8')})