import platform
import types
import unittest
import six
from apitools.base.protorpclite import descriptor
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.protorpclite import test_util
def testMessageField(self):
    field = messages.MessageField(descriptor.FieldDescriptor, 10)
    field.name = 'a_field'
    expected = descriptor.FieldDescriptor()
    expected.name = 'a_field'
    expected.number = 10
    expected.label = descriptor.FieldDescriptor.Label.OPTIONAL
    expected.variant = messages.MessageField.DEFAULT_VARIANT
    expected.type_name = 'apitools.base.protorpclite.descriptor.FieldDescriptor'
    described = descriptor.describe_field(field)
    described.check_initialized()
    self.assertEquals(expected, described)