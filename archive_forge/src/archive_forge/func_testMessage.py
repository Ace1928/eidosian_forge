import platform
import types
import unittest
import six
from apitools.base.protorpclite import descriptor
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.protorpclite import test_util
def testMessage(self):
    self.assertEquals(descriptor.describe_message(test_util.NestedMessage), descriptor.describe(test_util.NestedMessage))