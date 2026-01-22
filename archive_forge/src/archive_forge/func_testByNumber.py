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
def testByNumber(self):
    """Test look-up by number."""
    self.assertRaises(KeyError, Color.lookup_by_number, 'RED')
    self.assertEquals(Color.RED, Color.lookup_by_number(20))
    self.assertRaises(KeyError, Color.lookup_by_number, Color.RED)