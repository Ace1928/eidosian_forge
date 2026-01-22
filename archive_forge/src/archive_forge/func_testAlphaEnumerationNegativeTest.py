import datetime
import json
import unittest
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.protorpclite import protojson
from apitools.base.protorpclite import test_util
def testAlphaEnumerationNegativeTest(self):
    """The alpha enum value is invalid."""
    message = protojson.decode_message(MyMessage, '{"an_enum": "IAMINVALID"}')
    expected_message = MyMessage()
    self.assertEquals(expected_message, message)
    self.assertEquals('{"an_enum": "IAMINVALID"}', protojson.encode_message(message))