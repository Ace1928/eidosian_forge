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
def testEnumField_WrongType(self):
    """Test that forward referencing the wrong type raises an error."""
    global AMessage
    try:

        class AMessage(messages.Message):
            pass

        class AnotherMessage(messages.Message):
            a_field = messages.EnumField('AMessage', 1)
        self.assertRaises(messages.FieldDefinitionError, getattr, AnotherMessage.field_by_name('a_field'), 'type')
    finally:
        del AMessage