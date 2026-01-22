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
def testNoneAssignment(self):
    """Test that assigning None does not change comparison."""

    class MyMessage(messages.Message):
        my_field = messages.StringField(1)
    m1 = MyMessage()
    m2 = MyMessage()
    m2.my_field = None
    self.assertEquals(m1, m2)