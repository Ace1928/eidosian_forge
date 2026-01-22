import platform
import types
import unittest
import six
from apitools.base.protorpclite import descriptor
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.protorpclite import test_util
def testDateTimeField(self):
    field = message_types.DateTimeField(20)
    field.name = 'a_timestamp'
    expected = descriptor.FieldDescriptor()
    expected.name = 'a_timestamp'
    expected.number = 20
    expected.label = descriptor.FieldDescriptor.Label.OPTIONAL
    expected.variant = messages.MessageField.DEFAULT_VARIANT
    expected.type_name = 'apitools.base.protorpclite.message_types.DateTimeMessage'
    described = descriptor.describe_field(field)
    described.check_initialized()
    self.assertEquals(expected, described)