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
def testFindEnum(self):
    """Test that Enums are found."""

    class Color(messages.Enum):
        pass
    A = self.DefineMessage('a', 'A', {'Color': Color})
    self.assertEquals(Color, messages.find_definition('Color', A, importer=self.Importer))