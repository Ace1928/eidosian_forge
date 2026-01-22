import time
import uuid
from decimal import Decimal
from tests.unit import unittest
from boto.dynamodb.exceptions import DynamoDBKeyNotFoundError
from boto.dynamodb.exceptions import DynamoDBConditionalCheckFailedError
from boto.dynamodb.layer2 import Layer2
from boto.dynamodb.types import get_dynamodb_type, Binary
from boto.dynamodb.condition import BEGINS_WITH, CONTAINS, GT
from boto.compat import six, long_type
def test_large_integers(self):
    self.dynamodb.use_decimals()
    table = self.create_sample_table()
    item = table.new_item('foo', 'bar')
    item['decimalvalue'] = Decimal('129271300103398600')
    item.put()
    retrieved = table.get_item('foo', 'bar')
    self.assertEqual(retrieved['decimalvalue'], Decimal('129271300103398600'))
    self.assertEqual(retrieved['decimalvalue'], 129271300103398600)