import datetime
import json
import unittest
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.protorpclite import protojson
from apitools.base.protorpclite import test_util
def testDecodeNone(self):
    message = protojson.decode_message(MyMessage, '{"an_integer": []}')
    self.assertEquals(MyMessage(an_integer=None), message)