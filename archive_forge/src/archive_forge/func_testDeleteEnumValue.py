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
def testDeleteEnumValue(self):
    """Test that enum values cannot be deleted."""
    self.assertRaises(TypeError, delattr, Color, 'RED')