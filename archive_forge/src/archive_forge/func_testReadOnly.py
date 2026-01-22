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
def testReadOnly(self):
    """Test that objects are all read-only."""

    def action(field_class):
        field = field_class(10)
        self.assertRaises(AttributeError, setattr, field, 'number', 20)
        self.assertRaises(AttributeError, setattr, field, 'anything_else', 'whatever')
    self.ActionOnAllFieldClasses(action)