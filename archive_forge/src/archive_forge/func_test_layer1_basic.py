import time
import base64
from tests.unit import unittest
from boto.dynamodb.exceptions import DynamoDBKeyNotFoundError
from boto.dynamodb.exceptions import DynamoDBConditionalCheckFailedError
from boto.dynamodb.exceptions import DynamoDBValidationError
from boto.dynamodb.layer1 import Layer1
def test_layer1_basic(self):
    print('--- running DynamoDB Layer1 tests ---')
    c = self.dynamodb
    table_name = self.table_name
    hash_key_name = self.hash_key_name
    hash_key_type = self.hash_key_type
    range_key_name = self.range_key_name
    range_key_type = self.range_key_type
    read_units = self.read_units
    write_units = self.write_units
    schema = self.schema
    provisioned_throughput = self.provisioned_throughput
    result = self.create_table(table_name, schema, provisioned_throughput)
    assert result['TableDescription']['TableName'] == table_name
    result_schema = result['TableDescription']['KeySchema']
    assert result_schema['HashKeyElement']['AttributeName'] == hash_key_name
    assert result_schema['HashKeyElement']['AttributeType'] == hash_key_type
    assert result_schema['RangeKeyElement']['AttributeName'] == range_key_name
    assert result_schema['RangeKeyElement']['AttributeType'] == range_key_type
    result_thruput = result['TableDescription']['ProvisionedThroughput']
    assert result_thruput['ReadCapacityUnits'] == read_units
    assert result_thruput['WriteCapacityUnits'] == write_units
    result = c.describe_table(table_name)
    while result['Table']['TableStatus'] != 'ACTIVE':
        time.sleep(5)
        result = c.describe_table(table_name)
    result = c.list_tables()
    assert table_name in result['TableNames']
    new_read_units = 10
    new_write_units = 5
    new_provisioned_throughput = {'ReadCapacityUnits': new_read_units, 'WriteCapacityUnits': new_write_units}
    result = c.update_table(table_name, new_provisioned_throughput)
    result = c.describe_table(table_name)
    while result['Table']['TableStatus'] == 'UPDATING':
        time.sleep(5)
        result = c.describe_table(table_name)
    result_thruput = result['Table']['ProvisionedThroughput']
    assert result_thruput['ReadCapacityUnits'] == new_read_units
    assert result_thruput['WriteCapacityUnits'] == new_write_units
    item1_key = 'Amazon DynamoDB'
    item1_range = 'DynamoDB Thread 1'
    item1_data = {hash_key_name: {hash_key_type: item1_key}, range_key_name: {range_key_type: item1_range}, 'Message': {'S': 'DynamoDB thread 1 message text'}, 'LastPostedBy': {'S': 'User A'}, 'Views': {'N': '0'}, 'Replies': {'N': '0'}, 'Answered': {'N': '0'}, 'Tags': {'SS': ['index', 'primarykey', 'table']}, 'LastPostDateTime': {'S': '12/9/2011 11:36:03 PM'}}
    result = c.put_item(table_name, item1_data)
    key1 = {'HashKeyElement': {hash_key_type: item1_key}, 'RangeKeyElement': {range_key_type: item1_range}}
    result = c.get_item(table_name, key=key1, consistent_read=True)
    for name in item1_data:
        assert name in result['Item']
    invalid_key = {'HashKeyElement': {hash_key_type: 'bogus_key'}, 'RangeKeyElement': {range_key_type: item1_range}}
    self.assertRaises(DynamoDBKeyNotFoundError, c.get_item, table_name, key=invalid_key)
    attributes = ['Message', 'Views']
    result = c.get_item(table_name, key=key1, consistent_read=True, attributes_to_get=attributes)
    for name in result['Item']:
        assert name in attributes
    expected = {'Views': {'Value': {'N': '1'}}}
    self.assertRaises(DynamoDBConditionalCheckFailedError, c.delete_item, table_name, key=key1, expected=expected)
    attribute_updates = {'Views': {'Value': {'N': '5'}, 'Action': 'PUT'}, 'Tags': {'Value': {'SS': ['foobar']}, 'Action': 'ADD'}}
    result = c.update_item(table_name, key=key1, attribute_updates=attribute_updates)
    item_size_overflow_text = 'Text to be padded'.zfill(64 * 1024 - 32)
    attribute_updates = {'Message': {'Value': {'S': item_size_overflow_text}, 'Action': 'PUT'}}
    self.assertRaises(DynamoDBValidationError, c.update_item, table_name, key=key1, attribute_updates=attribute_updates)
    item2_key = 'Amazon DynamoDB'
    item2_range = 'DynamoDB Thread 2'
    item2_data = {hash_key_name: {hash_key_type: item2_key}, range_key_name: {range_key_type: item2_range}, 'Message': {'S': 'DynamoDB thread 2 message text'}, 'LastPostedBy': {'S': 'User A'}, 'Views': {'N': '0'}, 'Replies': {'N': '0'}, 'Answered': {'N': '0'}, 'Tags': {'SS': ['index', 'primarykey', 'table']}, 'LastPostDateTime': {'S': '12/9/2011 11:36:03 PM'}}
    result = c.put_item(table_name, item2_data)
    key2 = {'HashKeyElement': {hash_key_type: item2_key}, 'RangeKeyElement': {range_key_type: item2_range}}
    item3_key = 'Amazon S3'
    item3_range = 'S3 Thread 1'
    item3_data = {hash_key_name: {hash_key_type: item3_key}, range_key_name: {range_key_type: item3_range}, 'Message': {'S': 'S3 Thread 1 message text'}, 'LastPostedBy': {'S': 'User A'}, 'Views': {'N': '0'}, 'Replies': {'N': '0'}, 'Answered': {'N': '0'}, 'Tags': {'SS': ['largeobject', 'multipart upload']}, 'LastPostDateTime': {'S': '12/9/2011 11:36:03 PM'}}
    result = c.put_item(table_name, item3_data)
    key3 = {'HashKeyElement': {hash_key_type: item3_key}, 'RangeKeyElement': {range_key_type: item3_range}}
    result = c.query(table_name, {'S': 'Amazon DynamoDB'}, {'AttributeValueList': [{'S': 'DynamoDB'}], 'ComparisonOperator': 'BEGINS_WITH'})
    assert 'Count' in result
    assert result['Count'] == 2
    result = c.scan(table_name, {'Tags': {'AttributeValueList': [{'S': 'table'}], 'ComparisonOperator': 'CONTAINS'}})
    assert 'Count' in result
    assert result['Count'] == 2
    result = c.delete_item(table_name, key=key1)
    result = c.delete_item(table_name, key=key2)
    result = c.delete_item(table_name, key=key3)
    print('--- tests completed ---')