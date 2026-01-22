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
def testNestedFieldMapping(self):
    nested_msg = AdditionalPropertiesMessage()
    nested_msg.additionalProperties = [AdditionalPropertiesMessage.AdditionalProperty(key='key_one', value='value_one'), AdditionalPropertiesMessage.AdditionalProperty(key='key_two', value='value_two')]
    msg = HasNestedMessage(nested=nested_msg)
    encoded_msg = encoding.MessageToJson(msg)
    self.assertEqual({'nested': {'key_one': 'value_one', 'key_two': 'value_two'}}, json.loads(encoded_msg))
    new_msg = encoding.JsonToMessage(type(msg), encoded_msg)
    self.assertEqual(set(('key_one', 'key_two')), set([x.key for x in new_msg.nested.additionalProperties]))
    new_msg.nested.additionalProperties.pop()
    self.assertEqual(1, len(new_msg.nested.additionalProperties))
    self.assertEqual(2, len(msg.nested.additionalProperties))