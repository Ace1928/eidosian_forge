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
def testDictToAdditionalPropertyMessageSorted(self):
    tuples = [('key{0:02}'.format(i), 'value') for i in range(100)]
    dict_ = dict(tuples)
    encoded_msg = encoding.DictToAdditionalPropertyMessage(dict_, AdditionalPropertiesMessage, sort_items=True)
    expected_msg = AdditionalPropertiesMessage()
    expected_msg.additionalProperties = [AdditionalPropertiesMessage.AdditionalProperty(key=key, value=value) for key, value in tuples]
    self.assertEqual(encoded_msg, expected_msg)