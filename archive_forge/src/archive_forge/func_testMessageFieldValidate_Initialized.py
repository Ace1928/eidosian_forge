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
def testMessageFieldValidate_Initialized(self):
    """Test validation on message field."""

    class MyMessage(messages.Message):
        field1 = messages.IntegerField(1, required=True)
    field = messages.MessageField(MyMessage, 10)
    message = MyMessage()
    field.validate(message)
    message.field1 = 20
    field.validate(message)