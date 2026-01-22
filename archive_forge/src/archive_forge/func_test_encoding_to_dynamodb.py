from decimal import Decimal
from tests.compat import unittest
from boto.compat import six, json
from boto.dynamodb import types
from boto.dynamodb.exceptions import DynamoDBNumberError
def test_encoding_to_dynamodb(self):
    dynamizer = types.Dynamizer()
    self.assertEqual(dynamizer.encode('foo'), {'S': 'foo'})
    self.assertEqual(dynamizer.encode(54), {'N': '54'})
    self.assertEqual(dynamizer.encode(Decimal('1.1')), {'N': '1.1'})
    self.assertEqual(dynamizer.encode(set([1, 2, 3])), {'NS': ['1', '2', '3']})
    self.assertIn(dynamizer.encode(set(['foo', 'bar'])), ({'SS': ['foo', 'bar']}, {'SS': ['bar', 'foo']}))
    self.assertEqual(dynamizer.encode(types.Binary(b'\x01')), {'B': 'AQ=='})
    self.assertEqual(dynamizer.encode(set([types.Binary(b'\x01')])), {'BS': ['AQ==']})
    self.assertEqual(dynamizer.encode(['foo', 54, [1]]), {'L': [{'S': 'foo'}, {'N': '54'}, {'L': [{'N': '1'}]}]})
    self.assertEqual(dynamizer.encode({'foo': 'bar', 'hoge': {'sub': 1}}), {'M': {'foo': {'S': 'bar'}, 'hoge': {'M': {'sub': {'N': '1'}}}}})
    self.assertEqual(dynamizer.encode(None), {'NULL': True})
    self.assertEqual(dynamizer.encode(False), {'BOOL': False})