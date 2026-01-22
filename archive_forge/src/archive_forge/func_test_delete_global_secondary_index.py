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
def test_delete_global_secondary_index(self):
    with mock.patch.object(self.users.connection, 'update_table', return_value={}) as mock_update:
        self.users.delete_global_secondary_index('RandomGSIIndex')
    mock_update.assert_called_once_with('users', global_secondary_index_updates=[{'Delete': {'IndexName': 'RandomGSIIndex'}}])