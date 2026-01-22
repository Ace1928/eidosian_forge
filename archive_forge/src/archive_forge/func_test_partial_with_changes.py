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
def test_partial_with_changes(self):
    self.table.schema = [HashKey('username')]
    with mock.patch.object(self.table, '_update_item', return_value=True) as mock_update_item:
        self.johndoe.mark_clean()
        self.johndoe['first_name'] = 'J'
        self.johndoe['last_name'] = 'Doe'
        del self.johndoe['date_joined']
        self.assertTrue(self.johndoe.partial_save())
        self.assertFalse(self.johndoe.needs_save())
    self.assertTrue(mock_update_item.called)
    mock_update_item.assert_called_once_with({'username': 'johndoe'}, {'first_name': {'Action': 'PUT', 'Value': {'S': 'J'}}, 'last_name': {'Action': 'PUT', 'Value': {'S': 'Doe'}}, 'date_joined': {'Action': 'DELETE'}}, expects={'first_name': {'Value': {'S': 'John'}, 'Exists': True}, 'last_name': {'Exists': False}, 'date_joined': {'Value': {'N': '12345'}, 'Exists': True}})