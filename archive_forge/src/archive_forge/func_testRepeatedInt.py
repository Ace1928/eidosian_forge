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
def testRepeatedInt(self):
    """Test duplicated numbers are forbidden."""
    self.assertRaises(messages.EnumDefinitionError, messages.Enum.def_enum, {'Ok': 1, 'Repeated': 1}, 'BadEnum')