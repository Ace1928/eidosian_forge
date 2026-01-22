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
def testListAssignmentDoesNotCopy(self):

    class SimpleMessage(messages.Message):
        repeated = messages.IntegerField(1, repeated=True)
    message = SimpleMessage()
    original = message.repeated
    message.repeated = []
    self.assertFalse(original is message.repeated)