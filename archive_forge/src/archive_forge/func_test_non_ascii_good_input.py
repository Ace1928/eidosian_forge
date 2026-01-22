from decimal import Decimal
from tests.compat import unittest
from boto.compat import six, json
from boto.dynamodb import types
from boto.dynamodb.exceptions import DynamoDBNumberError
def test_non_ascii_good_input(self):
    data = types.Binary(b'\x88')
    self.assertEqual(b'\x88', data)
    self.assertEqual(b'\x88', bytes(data))