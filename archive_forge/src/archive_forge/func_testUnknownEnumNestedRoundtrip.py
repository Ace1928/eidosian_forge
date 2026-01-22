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
def testUnknownEnumNestedRoundtrip(self):
    json_with_typo = '{"outer_key": {"key_one": {"field_one": "VALUE_OEN", "field_two": "VALUE_OEN"}}}'
    msg = encoding.JsonToMessage(NestedAdditionalPropertiesWithEnumMessage, json_with_typo)
    self.assertEqual(json.loads(json_with_typo), json.loads(encoding.MessageToJson(msg)))