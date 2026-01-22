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
def test_create_global_secondary_index(self):
    with mock.patch.object(self.users.connection, 'update_table', return_value={}) as mock_update:
        self.users.create_global_secondary_index(global_index=GlobalAllIndex('JustCreatedIndex', parts=[HashKey('requiredHashKey')], throughput={'read': 2, 'write': 2}))
    mock_update.assert_called_once_with('users', global_secondary_index_updates=[{'Create': {'IndexName': 'JustCreatedIndex', 'KeySchema': [{'KeyType': 'HASH', 'AttributeName': 'requiredHashKey'}], 'Projection': {'ProjectionType': 'ALL'}, 'ProvisionedThroughput': {'WriteCapacityUnits': 2, 'ReadCapacityUnits': 2}}}], attribute_definitions=[{'AttributeName': 'requiredHashKey', 'AttributeType': 'S'}])