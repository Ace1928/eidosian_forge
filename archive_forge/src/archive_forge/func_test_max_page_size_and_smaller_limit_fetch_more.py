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
def test_max_page_size_and_smaller_limit_fetch_more(self):
    self.results = ResultSet(max_page_size=10)
    self.results.to_call(self.result_function, 'john', greeting='Hello', limit=5)
    self.results.fetch_more()
    self.result_function.assert_called_with('john', greeting='Hello', limit=5)
    self.result_function.reset_mock()