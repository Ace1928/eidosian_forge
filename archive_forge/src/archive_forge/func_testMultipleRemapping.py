import base64
import datetime
import json
import sys
import unittest
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.protorpclite import util
from apitools.base.py import encoding
from apitools.base.py import exceptions
from apitools.base.py import extra_types
def testMultipleRemapping(self):
    msg = MessageWithRemappings(double_encoding=MessageWithRemappings.SomeEnum.enum_value)
    json_message = encoding.MessageToJson(msg)
    self.assertEqual('{"doubleEncoding": "wire_name"}', json_message)
    self.assertEqual(msg, encoding.JsonToMessage(MessageWithRemappings, json_message))