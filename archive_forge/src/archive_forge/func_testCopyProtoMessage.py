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
def testCopyProtoMessage(self):
    msg = SimpleMessage(field='abc')
    new_msg = encoding.CopyProtoMessage(msg)
    self.assertEqual(msg.field, new_msg.field)
    msg.field = 'def'
    self.assertNotEqual(msg.field, new_msg.field)