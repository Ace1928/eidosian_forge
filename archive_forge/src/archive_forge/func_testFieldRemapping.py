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
def testFieldRemapping(self):
    msg = MessageWithRemappings(another_field='abc')
    json_message = encoding.MessageToJson(msg)
    self.assertEqual('{"anotherField": "abc"}', json_message)
    self.assertEqual(msg, encoding.JsonToMessage(MessageWithRemappings, json_message))