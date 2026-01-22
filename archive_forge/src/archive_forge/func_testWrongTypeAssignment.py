import datetime
import json
import unittest
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.protorpclite import protojson
from apitools.base.protorpclite import test_util
def testWrongTypeAssignment(self):
    """Test when wrong type is assigned to a field."""
    self.assertRaises(messages.ValidationError, protojson.decode_message, MyMessage, '{"a_string": 10}')
    self.assertRaises(messages.ValidationError, protojson.decode_message, MyMessage, '{"an_integer": 10.2}')
    self.assertRaises(messages.ValidationError, protojson.decode_message, MyMessage, '{"an_integer": "10.2"}')