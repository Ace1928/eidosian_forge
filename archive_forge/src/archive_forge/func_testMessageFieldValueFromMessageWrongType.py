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
def testMessageFieldValueFromMessageWrongType(self):

    class MyMessage(messages.Message):
        pass

    class HasMessage(messages.Message):
        field = messages.MessageField(MyMessage, 1)
    self.assertRaisesWithRegexpMatch(messages.DecodeError, 'Expected type MyMessage, got int: 10', HasMessage.field.value_from_message, 10)