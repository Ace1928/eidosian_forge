from decimal import Decimal
from tests.compat import unittest
from boto.compat import six, json
from boto.dynamodb import types
from boto.dynamodb.exceptions import DynamoDBNumberError
def test_decoding_to_dynamodb(self):
    dynamizer = types.Dynamizer()
    self.assertEqual(dynamizer.decode({'S': 'foo'}), 'foo')
    self.assertEqual(dynamizer.decode({'N': '54'}), 54)
    self.assertEqual(dynamizer.decode({'N': '1.1'}), Decimal('1.1'))
    self.assertEqual(dynamizer.decode({'NS': ['1', '2', '3']}), set([1, 2, 3]))
    self.assertEqual(dynamizer.decode({'SS': ['foo', 'bar']}), set(['foo', 'bar']))
    self.assertEqual(dynamizer.decode({'B': 'AQ=='}), types.Binary(b'\x01'))
    self.assertEqual(dynamizer.decode({'BS': ['AQ==']}), set([types.Binary(b'\x01')]))
    self.assertEqual(dynamizer.decode({'L': [{'S': 'foo'}, {'N': '54'}, {'L': [{'N': '1'}]}]}), ['foo', 54, [1]])
    self.assertEqual(dynamizer.decode({'M': {'foo': {'S': 'bar'}, 'hoge': {'M': {'sub': {'N': '1'}}}}}), {'foo': 'bar', 'hoge': {'sub': 1}})
    self.assertEqual(dynamizer.decode({'NULL': True}), None)
    self.assertEqual(dynamizer.decode({'BOOL': False}), False)