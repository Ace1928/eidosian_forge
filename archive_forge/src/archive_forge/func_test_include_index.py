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
def test_include_index(self):
    include_index = IncludeIndex('IncludeKeys', parts=[HashKey('username'), RangeKey('date_joined')], includes=['gender', 'friend_count'])
    self.assertEqual(include_index.name, 'IncludeKeys')
    self.assertEqual([part.attr_type for part in include_index.parts], ['HASH', 'RANGE'])
    self.assertEqual(include_index.projection_type, 'INCLUDE')
    self.assertEqual(include_index.definition(), [{'AttributeName': 'username', 'AttributeType': 'S'}, {'AttributeName': 'date_joined', 'AttributeType': 'S'}])
    self.assertEqual(include_index.schema(), {'IndexName': 'IncludeKeys', 'KeySchema': [{'AttributeName': 'username', 'KeyType': 'HASH'}, {'AttributeName': 'date_joined', 'KeyType': 'RANGE'}], 'Projection': {'ProjectionType': 'INCLUDE', 'NonKeyAttributes': ['gender', 'friend_count']}})