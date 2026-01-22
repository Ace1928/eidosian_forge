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
def test_batch_write_flushing(self):
    with mock.patch.object(self.users.connection, 'batch_write_item', return_value={}) as mock_batch:
        with self.users.batch_write() as batch:
            batch.put_item(data={'username': 'jane', 'date_joined': 12342547})
            batch.delete_item(username='johndoe1')
            batch.delete_item(username='johndoe2')
            batch.delete_item(username='johndoe3')
            batch.delete_item(username='johndoe4')
            batch.delete_item(username='johndoe5')
            batch.delete_item(username='johndoe6')
            batch.delete_item(username='johndoe7')
            batch.delete_item(username='johndoe8')
            batch.delete_item(username='johndoe9')
            batch.delete_item(username='johndoe10')
            batch.delete_item(username='johndoe11')
            batch.delete_item(username='johndoe12')
            batch.delete_item(username='johndoe13')
            batch.delete_item(username='johndoe14')
            batch.delete_item(username='johndoe15')
            batch.delete_item(username='johndoe16')
            batch.delete_item(username='johndoe17')
            batch.delete_item(username='johndoe18')
            batch.delete_item(username='johndoe19')
            batch.delete_item(username='johndoe20')
            batch.delete_item(username='johndoe21')
            batch.delete_item(username='johndoe22')
            batch.delete_item(username='johndoe23')
            self.assertEqual(mock_batch.call_count, 0)
            batch.delete_item(username='johndoe24')
            self.assertEqual(mock_batch.call_count, 1)
            batch.delete_item(username='johndoe25')
    self.assertEqual(mock_batch.call_count, 2)