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
@unittest.skipIf(six.PY3, 'skipping lossy_float_conversion test for Python 3.x')
def test_lossy_float_conversion(self):
    table = self.create_sample_table()
    item = table.new_item('foo', 'bar')
    item['floatvalue'] = 1.12345678912345
    item.put()
    retrieved = table.get_item('foo', 'bar')['floatvalue']
    self.assertNotEqual(1.12345678912345, retrieved)
    self.assertEqual(1.12345678912, retrieved)