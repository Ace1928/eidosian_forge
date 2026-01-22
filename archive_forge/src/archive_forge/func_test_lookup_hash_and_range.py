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
def test_lookup_hash_and_range(self):
    """Test the "lookup" function with a hash and range key"""
    expected = {'Item': {'username': {'S': 'johndoe'}, 'first_name': {'S': 'John'}, 'last_name': {'S': 'Doe'}, 'date_joined': {'N': '1366056668'}, 'friend_count': {'N': '3'}, 'friends': {'SS': ['alice', 'bob', 'jane']}}}
    self.users.schema = [HashKey('username'), RangeKey('date_joined', data_type=NUMBER)]
    with mock.patch.object(self.users, 'get_item', return_value=expected) as mock_get_item:
        self.users.lookup('johndoe', 1366056668)
    mock_get_item.assert_called_once_with(username='johndoe', date_joined=1366056668)