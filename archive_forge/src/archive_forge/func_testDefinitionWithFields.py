import platform
import types
import unittest
import six
from apitools.base.protorpclite import descriptor
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.protorpclite import test_util
def testDefinitionWithFields(self):

    class MessageWithFields(messages.Message):
        field1 = messages.IntegerField(10)
        field2 = messages.StringField(30)
        field3 = messages.IntegerField(20)
    expected = descriptor.MessageDescriptor()
    expected.name = 'MessageWithFields'
    expected.fields = [descriptor.describe_field(MessageWithFields.field_by_name('field1')), descriptor.describe_field(MessageWithFields.field_by_name('field3')), descriptor.describe_field(MessageWithFields.field_by_name('field2'))]
    described = descriptor.describe_message(MessageWithFields)
    described.check_initialized()
    self.assertEquals(expected, described)