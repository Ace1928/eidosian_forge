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
def testEnumField(self):
    """Test the construction of enum fields."""
    self.assertRaises(messages.FieldDefinitionError, messages.EnumField, str, 10)
    self.assertRaises(messages.FieldDefinitionError, messages.EnumField, messages.Enum, 10)

    class Color(messages.Enum):
        RED = 1
        GREEN = 2
        BLUE = 3
    field = messages.EnumField(Color, 10)
    self.assertEquals(Color, field.type)

    class Another(messages.Enum):
        VALUE = 1
    self.assertRaises(messages.InvalidDefaultError, messages.EnumField, Color, 10, default=Another.VALUE)