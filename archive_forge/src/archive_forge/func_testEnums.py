import platform
import types
import unittest
import six
from apitools.base.protorpclite import descriptor
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.protorpclite import test_util
def testEnums(self):
    """Test that enums are described."""
    module = self.LoadModule('my.package', 'class Enum1(messages.Enum): pass\nclass Enum2(messages.Enum): pass\n')
    enum1 = descriptor.EnumDescriptor()
    enum1.name = 'Enum1'
    enum2 = descriptor.EnumDescriptor()
    enum2.name = 'Enum2'
    expected = descriptor.FileDescriptor()
    expected.package = 'my.package'
    expected.enum_types = [enum1, enum2]
    described = descriptor.describe_file(module)
    described.check_initialized()
    self.assertEquals(expected, described)