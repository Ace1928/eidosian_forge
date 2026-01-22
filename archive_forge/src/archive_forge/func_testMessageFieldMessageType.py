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
def testMessageFieldMessageType(self):
    """Test message_type property."""

    class MyMessage(messages.Message):
        pass

    class HasMessage(messages.Message):
        field = messages.MessageField(MyMessage, 1)
    self.assertEqual(HasMessage.field.type, HasMessage.field.message_type)