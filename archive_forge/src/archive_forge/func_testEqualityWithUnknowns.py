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
def testEqualityWithUnknowns(self):
    """Test message class equality with unknown fields."""

    class MyMessage(messages.Message):
        field1 = messages.IntegerField(1)
    message1 = MyMessage()
    message2 = MyMessage()
    self.assertEquals(message1, message2)
    message1.set_unrecognized_field('unknown1', 'value1', messages.Variant.STRING)
    self.assertEquals(message1, message2)
    message1.set_unrecognized_field('unknown2', ['asdf', 3], messages.Variant.STRING)
    message1.set_unrecognized_field('unknown3', 4.7, messages.Variant.DOUBLE)
    self.assertEquals(message1, message2)