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
def test_delete_item_conditionally(self):
    with mock.patch.object(self.users.connection, 'delete_item', return_value={}) as mock_delete_item:
        self.assertTrue(self.users.delete_item(expected={'balance__eq': 0}, username='johndoe', date_joined=23456))
    mock_delete_item.assert_called_once_with('users', {'username': {'S': 'johndoe'}, 'date_joined': {'N': '23456'}}, expected={'balance': {'ComparisonOperator': 'EQ', 'AttributeValueList': [{'N': '0'}]}}, conditional_operator=None)

    def side_effect(*args, **kwargs):
        raise exceptions.ConditionalCheckFailedException(400, '', {})
    with mock.patch.object(self.users.connection, 'delete_item', side_effect=side_effect) as mock_delete_item:
        self.assertFalse(self.users.delete_item(expected={'balance__eq': 0}, username='johndoe', date_joined=23456))