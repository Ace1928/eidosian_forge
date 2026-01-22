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
def test_save_with_changes(self):
    with mock.patch.object(self.table, '_put_item', return_value=True) as mock_put_item:
        self.johndoe.mark_clean()
        self.johndoe['first_name'] = 'J'
        self.johndoe['new_attr'] = 'never_seen_before'
        self.assertTrue(self.johndoe.save())
        self.assertFalse(self.johndoe.needs_save())
    self.assertTrue(mock_put_item.called)
    mock_put_item.assert_called_once_with({'username': {'S': 'johndoe'}, 'first_name': {'S': 'J'}, 'new_attr': {'S': 'never_seen_before'}, 'date_joined': {'N': '12345'}}, expects={'username': {'Value': {'S': 'johndoe'}, 'Exists': True}, 'first_name': {'Value': {'S': 'John'}, 'Exists': True}, 'new_attr': {'Exists': False}, 'date_joined': {'Value': {'N': '12345'}, 'Exists': True}})