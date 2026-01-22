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
def testAllowNestedMessages(self):
    """Test allowing nested messages in a message definition."""

    class Trade(messages.Message):

        class Lot(messages.Message):
            pass

        class Agent(messages.Message):
            pass
    self.assertEquals(['Agent', 'Lot'], Trade.__messages__)
    self.assertEquals(Trade, Trade.Agent.message_definition())
    self.assertEquals(Trade, Trade.Lot.message_definition())

    def action():

        class Trade(messages.Message):
            NiceTry = messages.Message
    self.assertRaises(messages.MessageDefinitionError, action)