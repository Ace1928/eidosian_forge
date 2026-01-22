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
def testValidate_Required(self):
    """Tests validation of required fields."""

    class SimpleMessage(messages.Message):
        required = messages.IntegerField(1, required=True)
    simple_message = SimpleMessage()
    self.assertRaises(messages.ValidationError, simple_message.check_initialized)
    simple_message.required = 10
    simple_message.check_initialized()