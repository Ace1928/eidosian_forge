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
def testFieldByName(self):
    """Test getting field by name."""
    ComplexMessage = self.CreateMessageClass()
    self.assertEquals(3, ComplexMessage.field_by_name('a3').number)
    self.assertEquals(1, ComplexMessage.field_by_name('b1').number)
    self.assertEquals(2, ComplexMessage.field_by_name('c2').number)
    self.assertRaises(KeyError, ComplexMessage.field_by_name, 'unknown')