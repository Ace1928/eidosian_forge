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
def testMessageDefinition(self):
    """Test that enumeration knows its enclosing message definition."""

    class OuterMessage(messages.Message):

        class InnerMessage(messages.Message):
            pass
    self.assertEquals(None, OuterMessage.message_definition())
    self.assertEquals(OuterMessage, OuterMessage.InnerMessage.message_definition())