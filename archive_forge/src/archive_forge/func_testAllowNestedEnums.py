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
def testAllowNestedEnums(self):
    """Test allowing nested enums in a message definition."""

    class Trade(messages.Message):

        class Duration(messages.Enum):
            GTC = 1
            DAY = 2

        class Currency(messages.Enum):
            USD = 1
            GBP = 2
            INR = 3
    self.assertEquals(['Currency', 'Duration'], Trade.__enums__)
    self.assertEquals(Trade, Trade.Duration.message_definition())