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
def testDictToAdditionalPropertyMessageNumeric(self):
    dict_ = {'key': 1}
    encoded_msg = encoding.DictToAdditionalPropertyMessage(dict_, AdditionalIntPropertiesMessage)
    expected_msg = AdditionalIntPropertiesMessage()
    expected_msg.additionalProperties = [AdditionalIntPropertiesMessage.AdditionalProperty(key='key', value=1)]
    self.assertEqual(encoded_msg, expected_msg)