import datetime
import json
import unittest
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.protorpclite import protojson
from apitools.base.protorpclite import test_util
def testDecode(self):
    self.assertEqual(MyMessage(a_string='{decoded}xyz'), self.protojson.decode_message(MyMessage, '{"a_string": "xyz"}'))