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
def testGlobalFind(self):
    """Test finding definitions from fully qualified module names."""
    A = self.DefineMessage('a.b.c', 'A', {})
    self.assertEquals(A, messages.find_definition('a.b.c.A', importer=self.Importer))
    B = self.DefineMessage('a.b.c', 'B', {'C': {}})
    self.assertEquals(B.C, messages.find_definition('a.b.c.B.C', importer=self.Importer))