from decimal import Decimal
from tests.compat import unittest
from boto.compat import six, json
from boto.dynamodb import types
from boto.dynamodb.exceptions import DynamoDBNumberError
@unittest.skipUnless(six.PY3, 'Python 3 only')
def test_unicode_py3(self):
    with self.assertRaises(TypeError):
        types.Binary(u'\x01')