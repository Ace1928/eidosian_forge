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
def testValidate_None(self):
    """Test that None is valid for non-required fields."""

    def action(field_class):
        field = field_class(1)
        field.validate(None)
        field = field_class(1, required=True)
        self.assertRaisesWithRegexpMatch(messages.ValidationError, 'Required field is missing', field.validate, None)
        field = field_class(1, repeated=True)
        field.validate(None)
        self.assertRaisesWithRegexpMatch(messages.ValidationError, 'Repeated values for %s may not be None' % field_class.__name__, field.validate, [None])
        self.assertRaises(messages.ValidationError, field.validate, (None,))
    self.ActionOnAllFieldClasses(action)