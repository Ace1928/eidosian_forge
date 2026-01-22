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
def testDefinitionName_Nested(self):
    """Test nested message names."""

    class MyMessage(messages.Message):

        class NestedMessage(messages.Message):

            class NestedMessage(messages.Message):
                pass
    module_name = test_util.get_module_name(MessageTest)
    self.assertEquals('%s.MyMessage.NestedMessage' % module_name, MyMessage.NestedMessage.definition_name())
    self.assertEquals('%s.MyMessage' % module_name, MyMessage.NestedMessage.outer_definition_name())
    self.assertEquals(module_name, MyMessage.NestedMessage.definition_package())
    self.assertEquals('%s.MyMessage.NestedMessage.NestedMessage' % module_name, MyMessage.NestedMessage.NestedMessage.definition_name())
    self.assertEquals('%s.MyMessage.NestedMessage' % module_name, MyMessage.NestedMessage.NestedMessage.outer_definition_name())
    self.assertEquals(module_name, MyMessage.NestedMessage.NestedMessage.definition_package())