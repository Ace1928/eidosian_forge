import pickle
import re
import sys
import types
import unittest
import six
from apitools.base.protorpclite import descriptor
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.protorpclite import test_util
def testGetUnsetRepeatedValue(self):

    class SomeMessage(messages.Message):
        repeated = messages.IntegerField(1, repeated=True)
    instance = SomeMessage()
    self.assertEquals([], instance.repeated)
    self.assertTrue(isinstance(instance.repeated, messages.FieldList))