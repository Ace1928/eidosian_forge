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
def testNestedMethodsNotAllowed(self):
    """Test that method definitions on Message classes are not allowed."""

    def action():

        class WithMethods(messages.Message):

            def not_allowed(self):
                pass
    self.assertRaises(messages.MessageDefinitionError, action)