from tests.compat import mock, unittest
from boto.dynamodb2 import exceptions
from boto.dynamodb2.fields import (HashKey, RangeKey,
from boto.dynamodb2.items import Item
from boto.dynamodb2.layer1 import DynamoDBConnection
from boto.dynamodb2.results import ResultSet, BatchGetResultSet
from boto.dynamodb2.table import Table
from boto.dynamodb2.types import (STRING, NUMBER, BINARY,
from boto.exception import JSONResponseError
from boto.compat import six, long_type
def test_global_keys_only_index(self):
    keys_only = GlobalKeysOnlyIndex('KeysOnly', parts=[HashKey('username'), RangeKey('date_joined')], throughput={'read': 3, 'write': 4})
    self.assertEqual(keys_only.name, 'KeysOnly')
    self.assertEqual([part.attr_type for part in keys_only.parts], ['HASH', 'RANGE'])
    self.assertEqual(keys_only.projection_type, 'KEYS_ONLY')
    self.assertEqual(keys_only.definition(), [{'AttributeName': 'username', 'AttributeType': 'S'}, {'AttributeName': 'date_joined', 'AttributeType': 'S'}])
    self.assertEqual(keys_only.schema(), {'IndexName': 'KeysOnly', 'KeySchema': [{'AttributeName': 'username', 'KeyType': 'HASH'}, {'AttributeName': 'date_joined', 'KeyType': 'RANGE'}], 'Projection': {'ProjectionType': 'KEYS_ONLY'}, 'ProvisionedThroughput': {'ReadCapacityUnits': 3, 'WriteCapacityUnits': 4}})