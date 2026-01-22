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
def testUnrecognizedFieldIterAbortAfterFirstError(self):
    m = encoding.DictToMessage({'msg_field': {'field_one': 3}, 'enum_field': 3}, NestedWithEnumMessage)
    self.assertEqual(1, len(list(encoding.UnrecognizedFieldIter(m))))