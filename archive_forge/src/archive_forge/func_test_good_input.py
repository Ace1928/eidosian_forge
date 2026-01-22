from decimal import Decimal
from tests.compat import unittest
from boto.compat import six, json
from boto.dynamodb import types
from boto.dynamodb.exceptions import DynamoDBNumberError
def test_good_input(self):
    data = types.Binary(b'\x01')
    self.assertEqual(b'\x01', data)
    self.assertEqual(b'\x01', bytes(data))