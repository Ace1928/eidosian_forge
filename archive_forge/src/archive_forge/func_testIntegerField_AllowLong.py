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
def testIntegerField_AllowLong(self):
    """Test that the integer field allows for longs."""
    if six.PY2:
        messages.IntegerField(10, default=long(10))