import platform
import types
import unittest
import six
from apitools.base.protorpclite import descriptor
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.protorpclite import test_util
def testWithModules(self):
    """Test what happens when no modules provided."""
    modules = [types.ModuleType('package1'), types.ModuleType('package1')]
    file1 = descriptor.FileDescriptor()
    file1.package = 'package1'
    file2 = descriptor.FileDescriptor()
    file2.package = 'package2'
    expected = descriptor.FileSet()
    expected.files = [file1, file1]
    described = descriptor.describe_file_set(modules)
    described.check_initialized()
    self.assertEquals(expected, described)