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
def testEnumField_ForwardReference(self):
    """Test the construction of forward reference enum fields."""
    global MyMessage
    global ForwardEnum
    global ForwardMessage
    try:

        class MyMessage(messages.Message):
            forward = messages.EnumField('ForwardEnum', 1)
            nested = messages.EnumField('ForwardMessage.NestedEnum', 2)
            inner = messages.EnumField('Inner', 3)

            class Inner(messages.Enum):
                pass

        class ForwardEnum(messages.Enum):
            pass

        class ForwardMessage(messages.Message):

            class NestedEnum(messages.Enum):
                pass
        self.assertEquals(ForwardEnum, MyMessage.field_by_name('forward').type)
        self.assertEquals(ForwardMessage.NestedEnum, MyMessage.field_by_name('nested').type)
        self.assertEquals(MyMessage.Inner, MyMessage.field_by_name('inner').type)
    finally:
        try:
            del MyMessage
            del ForwardEnum
            del ForwardMessage
        except:
            pass