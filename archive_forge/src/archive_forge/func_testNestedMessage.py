import platform
import types
import unittest
import six
from apitools.base.protorpclite import descriptor
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.protorpclite import test_util
def testNestedMessage(self):

    class MessageWithMessage(messages.Message):

        class Nesty(messages.Message):
            pass
    expected = descriptor.MessageDescriptor()
    expected.name = 'MessageWithMessage'
    expected.message_types = [descriptor.describe_message(MessageWithMessage.Nesty)]
    described = descriptor.describe_message(MessageWithMessage)
    described.check_initialized()
    self.assertEquals(expected, described)