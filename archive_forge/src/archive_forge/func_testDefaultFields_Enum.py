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
def testDefaultFields_Enum(self):
    """Test the default for enum fields."""

    class Symbol(messages.Enum):
        ALPHA = 1
        BETA = 2
        GAMMA = 3
    field = messages.EnumField(Symbol, 1, default=Symbol.ALPHA)
    self.assertEquals(Symbol.ALPHA, field.default)