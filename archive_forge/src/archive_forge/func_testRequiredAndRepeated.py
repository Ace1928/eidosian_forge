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
def testRequiredAndRepeated(self):
    """Test setting the required and repeated fields."""

    def action(field_class):
        field_class(1, required=True)
        field_class(1, repeated=True)
        self.assertRaises(messages.FieldDefinitionError, field_class, 1, required=True, repeated=True)
    self.ActionOnAllFieldClasses(action)