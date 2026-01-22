import platform
import types
import unittest
import six
from apitools.base.protorpclite import descriptor
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.protorpclite import test_util
def testNestedEnum(self):

    class MessageWithEnum(messages.Message):

        class Mood(messages.Enum):
            GOOD = 1
            BAD = 2
            UGLY = 3

        class Music(messages.Enum):
            CLASSIC = 1
            JAZZ = 2
            BLUES = 3
    expected = descriptor.MessageDescriptor()
    expected.name = 'MessageWithEnum'
    expected.enum_types = [descriptor.describe_enum(MessageWithEnum.Mood), descriptor.describe_enum(MessageWithEnum.Music)]
    described = descriptor.describe_message(MessageWithEnum)
    described.check_initialized()
    self.assertEquals(expected, described)