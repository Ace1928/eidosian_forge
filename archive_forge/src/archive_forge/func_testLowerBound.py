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
def testLowerBound(self):
    """Test that zero is accepted by enum def."""

    class NotImportant(messages.Enum):
        """Testing for value zero"""
        VALUE = 0
    self.assertEquals(0, int(NotImportant.VALUE))