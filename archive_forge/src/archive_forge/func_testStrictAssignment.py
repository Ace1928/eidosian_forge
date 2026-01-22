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
def testStrictAssignment(self):
    """Tests that cannot assign to unknown or non-reserved attributes."""

    class SimpleMessage(messages.Message):
        field = messages.IntegerField(1)
    simple_message = SimpleMessage()
    self.assertRaises(AttributeError, setattr, simple_message, 'does_not_exist', 10)