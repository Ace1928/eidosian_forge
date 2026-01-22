import time
from tests.unit import unittest
from boto.dynamodb2 import exceptions
from boto.dynamodb2.layer1 import DynamoDBConnection
def test_integrated(self):
    result = self.create_table(self.table_name, self.attributes, self.schema, self.provisioned_throughput, self.lsi)
    self.assertEqual(result['TableDescription']['TableName'], self.table_name)
    description = self.dynamodb.describe_table(self.table_name)
    self.assertEqual(description['Table']['ItemCount'], 0)
    record_1_data = {'username': {'S': 'johndoe'}, 'first_name': {'S': 'John'}, 'last_name': {'S': 'Doe'}, 'date_joined': {'N': '1366056668'}, 'friend_count': {'N': '3'}, 'friends': {'SS': ['alice', 'bob', 'jane']}}
    r1_result = self.dynamodb.put_item(self.table_name, record_1_data)
    record_1 = self.dynamodb.get_item(self.table_name, key={'username': {'S': 'johndoe'}, 'date_joined': {'N': '1366056668'}}, consistent_read=True)
    self.assertEqual(record_1['Item']['username']['S'], 'johndoe')
    self.assertEqual(record_1['Item']['first_name']['S'], 'John')
    self.assertEqual(record_1['Item']['friends']['SS'], ['alice', 'bob', 'jane'])
    self.dynamodb.batch_write_item({self.table_name: [{'PutRequest': {'Item': {'username': {'S': 'jane'}, 'first_name': {'S': 'Jane'}, 'last_name': {'S': 'Doe'}, 'date_joined': {'N': '1366056789'}, 'friend_count': {'N': '1'}, 'friends': {'SS': ['johndoe']}}}}]})
    lsi_results = self.dynamodb.query(self.table_name, index_name='MostRecentIndex', key_conditions={'username': {'AttributeValueList': [{'S': 'johndoe'}], 'ComparisonOperator': 'EQ'}}, consistent_read=True)
    self.assertEqual(lsi_results['Count'], 1)
    results = self.dynamodb.query(self.table_name, key_conditions={'username': {'AttributeValueList': [{'S': 'jane'}], 'ComparisonOperator': 'EQ'}, 'date_joined': {'AttributeValueList': [{'N': '1366050000'}], 'ComparisonOperator': 'GT'}}, consistent_read=True)
    self.assertEqual(results['Count'], 1)
    results = self.dynamodb.scan(self.table_name)
    self.assertEqual(results['Count'], 2)
    s_items = sorted([res['username']['S'] for res in results['Items']])
    self.assertEqual(s_items, ['jane', 'johndoe'])
    self.dynamodb.delete_item(self.table_name, key={'username': {'S': 'johndoe'}, 'date_joined': {'N': '1366056668'}})
    results = self.dynamodb.scan(self.table_name)
    self.assertEqual(results['Count'], 1)
    self.dynamodb.batch_write_item({self.table_name: [{'PutRequest': {'Item': {'username': {'S': 'johndoe'}, 'first_name': {'S': 'Johann'}, 'last_name': {'S': 'Does'}, 'date_joined': {'N': '1366058000'}, 'friend_count': {'N': '1'}, 'friends': {'SS': ['jane']}}}, 'PutRequest': {'Item': {'username': {'S': 'alice'}, 'first_name': {'S': 'Alice'}, 'last_name': {'S': 'Expert'}, 'date_joined': {'N': '1366056800'}, 'friend_count': {'N': '2'}, 'friends': {'SS': ['johndoe', 'jane']}}}}]})
    time.sleep(20)
    results = self.dynamodb.scan(self.table_name, segment=0, total_segments=2)
    self.assertTrue(results['Count'] in [1, 2])
    results = self.dynamodb.scan(self.table_name, segment=1, total_segments=2)
    self.assertTrue(results['Count'] in [1, 2])