from decimal import Decimal
from tests.compat import unittest
from boto.compat import six, json
from boto.dynamodb import types
from boto.dynamodb.exceptions import DynamoDBNumberError
@unittest.skipUnless(six.PY2, 'Python 2 only')
def test_unicode_py2(self):
    data = types.Binary(u'\x01')
    self.assertEqual(data, b'\x01')
    self.assertEqual(bytes(data), b'\x01')
    self.assertEqual(data, u'\x01')
    self.assertEqual(type(data.value), bytes)