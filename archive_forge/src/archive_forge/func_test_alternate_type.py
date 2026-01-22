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
def test_alternate_type(self):
    alt_key = HashKey('alt', data_type=NUMBER)
    self.assertEqual(alt_key.name, 'alt')
    self.assertEqual(alt_key.data_type, NUMBER)
    self.assertEqual(alt_key.attr_type, 'HASH')
    self.assertEqual(alt_key.definition(), {'AttributeName': 'alt', 'AttributeType': 'N'})
    self.assertEqual(alt_key.schema(), {'AttributeName': 'alt', 'KeyType': 'HASH'})