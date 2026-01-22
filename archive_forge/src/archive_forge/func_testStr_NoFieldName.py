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
def testStr_NoFieldName(self):
    """Test string version of ValidationError when no name provided."""
    self.assertEquals('Validation error', str(messages.ValidationError('Validation error')))