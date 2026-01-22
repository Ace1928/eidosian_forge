import datetime
import sys
import types
import unittest
import six
from apitools.base.protorpclite import test_util
from apitools.base.protorpclite import util
def testHasPackage(self):
    module = self.CreateModule('service_module')
    module.package = 'my_package'
    self.assertPackageEquals('my_package', util.get_package_for_module(module))