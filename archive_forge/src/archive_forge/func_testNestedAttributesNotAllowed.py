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
def testNestedAttributesNotAllowed(self):
    """Test attribute assignment on Message classes is not allowed."""

    def int_attribute():

        class WithMethods(messages.Message):
            not_allowed = 1

    def string_attribute():

        class WithMethods(messages.Message):
            not_allowed = 'not allowed'

    def enum_attribute():

        class WithMethods(messages.Message):
            not_allowed = Color.RED
    for action in (int_attribute, string_attribute, enum_attribute):
        self.assertRaises(messages.MessageDefinitionError, action)