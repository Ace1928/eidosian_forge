import datetime
import json
import unittest
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.protorpclite import protojson
from apitools.base.protorpclite import test_util
def testNumericEnumeration(self):
    """Test that numbers work for enum values."""
    message = protojson.decode_message(MyMessage, '{"an_enum": 2}')
    expected_message = MyMessage()
    expected_message.an_enum = MyMessage.Color.GREEN
    self.assertEquals(expected_message, message)