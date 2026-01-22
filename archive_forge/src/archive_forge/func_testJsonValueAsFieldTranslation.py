import datetime
import json
import math
import unittest
from apitools.base.protorpclite import messages
from apitools.base.py import encoding
from apitools.base.py import exceptions
from apitools.base.py import extra_types
def testJsonValueAsFieldTranslation(self):

    class HasJsonValueMsg(messages.Message):
        some_value = messages.MessageField(extra_types.JsonValue, 1)
    msg_json = '{"some_value": [1, 2, 3]}'
    msg = HasJsonValueMsg(some_value=encoding.PyValueToMessage(extra_types.JsonValue, [1, 2, 3]))
    self.assertEqual(msg, encoding.JsonToMessage(HasJsonValueMsg, msg_json))
    self.assertEqual(msg_json, encoding.MessageToJson(msg))