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
def testNestedIncludeFields(self):
    msg = HasNestedMessage(nested=AdditionalPropertiesMessage(additionalProperties=[]))
    self.assertEqual('{"nested": null}', encoding.MessageToJson(msg, include_fields=['nested']))
    self.assertEqual('{"nested": {"additionalProperties": []}}', encoding.MessageToJson(msg, include_fields=['nested.additionalProperties']))
    msg = ExtraNestedMessage(nested=msg)
    self.assertEqual('{"nested": {"nested": null}}', encoding.MessageToJson(msg, include_fields=['nested.nested']))
    self.assertIn(encoding.MessageToJson(msg, include_fields=['nested.nested_list']), ['{"nested": {"nested": {}, "nested_list": []}}', '{"nested": {"nested_list": [], "nested": {}}}'])
    self.assertEqual('{"nested": {"nested": {"additionalProperties": []}}}', encoding.MessageToJson(msg, include_fields=['nested.nested.additionalProperties']))