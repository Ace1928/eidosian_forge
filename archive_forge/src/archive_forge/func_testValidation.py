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
def testValidation(self):
    """Test validation of message values."""

    class SubMessage(messages.Message):
        pass

    class Message(messages.Message):
        val = messages.MessageField(SubMessage, 1)
    message = Message()
    message_field = messages.MessageField(Message, 1)
    message_field.validate(message)
    message.val = SubMessage()
    message_field.validate(message)
    self.assertRaises(messages.ValidationError, setattr, message, 'val', [SubMessage()])

    class Message(messages.Message):
        val = messages.MessageField(SubMessage, 1, required=True)
    message = Message()
    message_field = messages.MessageField(Message, 1)
    message_field.validate(message)
    message.val = SubMessage()
    message_field.validate(message)
    self.assertRaises(messages.ValidationError, setattr, message, 'val', [SubMessage()])

    class Message(messages.Message):
        val = messages.MessageField(SubMessage, 1, repeated=True)
    message = Message()
    message_field = messages.MessageField(Message, 1)
    message_field.validate(message)
    self.assertRaisesWithRegexpMatch(messages.ValidationError, 'Field val is repeated. Found: <SubMessage>', setattr, message, 'val', SubMessage())
    message.val = [SubMessage()]
    message_field.validate(message)