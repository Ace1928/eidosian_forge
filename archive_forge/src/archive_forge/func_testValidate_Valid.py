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
def testValidate_Valid(self):
    """Test validation of valid values."""
    values = {messages.IntegerField: 10, messages.FloatField: 1.5, messages.BooleanField: False, messages.BytesField: b'abc', messages.StringField: u'abc'}

    def action(field_class):
        field = field_class(1)
        field.validate(values[field_class])
        field = field_class(1, required=True)
        field.validate(values[field_class])
        field = field_class(1, repeated=True)
        field.validate([])
        field.validate(())
        field.validate([values[field_class]])
        field.validate((values[field_class],))
        self.assertRaises(messages.ValidationError, field.validate, values[field_class])
        self.assertRaises(messages.ValidationError, field.validate, values[field_class])
    self.ActionOnAllFieldClasses(action)