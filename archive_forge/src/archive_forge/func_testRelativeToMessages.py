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
def testRelativeToMessages(self):
    """Test finding definitions relative to Message definitions."""
    A = self.DefineMessage('a.b', 'A', {'B': {'C': {}, 'D': {}}})
    B = A.B
    C = A.B.C
    D = A.B.D
    self.assertEquals(A, messages.find_definition('A', A, importer=self.Importer))
    self.assertEquals(B, messages.find_definition('B', A, importer=self.Importer))
    self.assertEquals(C, messages.find_definition('B.C', A, importer=self.Importer))
    self.assertEquals(D, messages.find_definition('B.D', A, importer=self.Importer))
    self.assertEquals(A, messages.find_definition('A', B, importer=self.Importer))
    self.assertEquals(B, messages.find_definition('B', B, importer=self.Importer))
    self.assertEquals(C, messages.find_definition('C', B, importer=self.Importer))
    self.assertEquals(D, messages.find_definition('D', B, importer=self.Importer))
    self.assertEquals(A, messages.find_definition('A', C, importer=self.Importer))
    self.assertEquals(B, messages.find_definition('B', C, importer=self.Importer))
    self.assertEquals(C, messages.find_definition('C', C, importer=self.Importer))
    self.assertEquals(D, messages.find_definition('D', C, importer=self.Importer))
    self.assertEquals(A, messages.find_definition('b.A', C, importer=self.Importer))
    self.assertEquals(B, messages.find_definition('b.A.B', C, importer=self.Importer))
    self.assertEquals(C, messages.find_definition('b.A.B.C', C, importer=self.Importer))
    self.assertEquals(D, messages.find_definition('b.A.B.D', C, importer=self.Importer))