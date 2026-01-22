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
def testNoDefinition(self):
    """Test not finding a definition in an existing module."""
    self.DefineModule('i.am.a.module')
    self.assertRaises(messages.DefinitionNotFoundError, messages.find_definition, 'i.am.a.module.MyMessage', importer=self.Importer)