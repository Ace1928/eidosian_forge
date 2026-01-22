import platform
import types
import unittest
import six
from apitools.base.protorpclite import descriptor
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.protorpclite import test_util
def testFindMessage(self):
    self.assertEquals(descriptor.describe_message(descriptor.FileSet), descriptor.import_descriptor_loader('apitools.base.protorpclite.descriptor.FileSet'))