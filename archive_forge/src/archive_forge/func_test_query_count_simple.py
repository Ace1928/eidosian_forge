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
def test_query_count_simple(self):
    expected_0 = {'Count': 0.0}
    expected_1 = {'Count': 10.0}
    with mock.patch.object(self.users.connection, 'query', return_value=expected_0) as mock_query:
        results = self.users.query_count(username__eq='notmyname')
        self.assertTrue(isinstance(results, int))
        self.assertEqual(results, 0)
    self.assertEqual(mock_query.call_count, 1)
    self.assertIn('scan_index_forward', mock_query.call_args[1])
    self.assertEqual(True, mock_query.call_args[1]['scan_index_forward'])
    self.assertIn('limit', mock_query.call_args[1])
    self.assertEqual(None, mock_query.call_args[1]['limit'])
    with mock.patch.object(self.users.connection, 'query', return_value=expected_1) as mock_query:
        results = self.users.query_count(username__gt='somename', consistent=True, scan_index_forward=False, limit=10)
        self.assertTrue(isinstance(results, int))
        self.assertEqual(results, 10)
    self.assertEqual(mock_query.call_count, 1)
    self.assertIn('scan_index_forward', mock_query.call_args[1])
    self.assertEqual(False, mock_query.call_args[1]['scan_index_forward'])
    self.assertIn('limit', mock_query.call_args[1])
    self.assertEqual(10, mock_query.call_args[1]['limit'])