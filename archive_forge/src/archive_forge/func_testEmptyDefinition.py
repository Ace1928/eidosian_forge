import platform
import types
import unittest
import six
from apitools.base.protorpclite import descriptor
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.protorpclite import test_util
def testEmptyDefinition(self):

    class MyMessage(messages.Message):
        pass
    expected = descriptor.MessageDescriptor()
    expected.name = 'MyMessage'
    described = descriptor.describe_message(MyMessage)
    described.check_initialized()
    self.assertEquals(expected, described)