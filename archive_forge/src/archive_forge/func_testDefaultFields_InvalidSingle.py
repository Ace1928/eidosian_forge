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
def testDefaultFields_InvalidSingle(self):
    """Test default field is correct type (invalid single)."""

    def action(field_class):
        self.assertRaises(messages.InvalidDefaultError, field_class, 1, default=object())
    self.ActionOnAllFieldClasses(action)