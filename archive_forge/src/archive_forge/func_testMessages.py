import platform
import types
import unittest
import six
from apitools.base.protorpclite import descriptor
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.protorpclite import test_util
def testMessages(self):
    """Test that messages are described."""
    module = self.LoadModule('my.package', 'class Message1(messages.Message): pass\nclass Message2(messages.Message): pass\n')
    message1 = descriptor.MessageDescriptor()
    message1.name = 'Message1'
    message2 = descriptor.MessageDescriptor()
    message2.name = 'Message2'
    expected = descriptor.FileDescriptor()
    expected.package = 'my.package'
    expected.message_types = [message1, message2]
    described = descriptor.describe_file(module)
    described.check_initialized()
    self.assertEquals(expected, described)