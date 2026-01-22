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
def testMessageField_ForwardReference(self):
    """Test the construction of forward reference message fields."""
    global MyMessage
    global ForwardMessage
    try:

        class MyMessage(messages.Message):
            self_reference = messages.MessageField('MyMessage', 1)
            forward = messages.MessageField('ForwardMessage', 2)
            nested = messages.MessageField('ForwardMessage.NestedMessage', 3)
            inner = messages.MessageField('Inner', 4)

            class Inner(messages.Message):
                sibling = messages.MessageField('Sibling', 1)

            class Sibling(messages.Message):
                pass

        class ForwardMessage(messages.Message):

            class NestedMessage(messages.Message):
                pass
        self.assertEquals(MyMessage, MyMessage.field_by_name('self_reference').type)
        self.assertEquals(ForwardMessage, MyMessage.field_by_name('forward').type)
        self.assertEquals(ForwardMessage.NestedMessage, MyMessage.field_by_name('nested').type)
        self.assertEquals(MyMessage.Inner, MyMessage.field_by_name('inner').type)
        self.assertEquals(MyMessage.Sibling, MyMessage.Inner.field_by_name('sibling').type)
    finally:
        try:
            del MyMessage
            del ForwardMessage
        except:
            pass