import datetime
import sys
import types
import unittest
import six
from apitools.base.protorpclite import test_util
from apitools.base.protorpclite import util
def testNoPackageAtAll(self):
    module = self.CreateModule('__main__')
    self.assertPackageEquals('__main__', util.get_package_for_module(module))