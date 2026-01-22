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
def testStringField_BadUnicodeInDefault(self):
    """Test binary values in string field."""
    self.assertRaisesWithRegexpMatch(messages.InvalidDefaultError, "Invalid default value for StringField:.*: Field encountered non-UTF-8 string .*: 'utf.?8' codec can't decode byte 0xc3 in position 0: invalid continuation byte", messages.StringField, 1, default=b'\xc3(')