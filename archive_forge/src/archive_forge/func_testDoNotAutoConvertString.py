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
def testDoNotAutoConvertString(self):
    string_field = messages.StringField(1, repeated=True)
    self.assertRaises(messages.ValidationError, messages.FieldList, string_field, 'abc')