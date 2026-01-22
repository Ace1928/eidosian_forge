import platform
import types
import unittest
import six
from apitools.base.protorpclite import descriptor
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.protorpclite import test_util
def testDefault_EnumField(self):

    class MyEnum(messages.Enum):
        VAL = 1
    module_name = test_util.get_module_name(MyEnum)
    field = messages.EnumField(MyEnum, 10, default=MyEnum.VAL)
    field.name = 'a_field'
    expected = descriptor.FieldDescriptor()
    expected.name = 'a_field'
    expected.number = 10
    expected.label = descriptor.FieldDescriptor.Label.OPTIONAL
    expected.variant = messages.EnumField.DEFAULT_VARIANT
    expected.type_name = '%s.MyEnum' % module_name
    expected.default_value = '1'
    described = descriptor.describe_field(field)
    self.assertEquals(expected, described)