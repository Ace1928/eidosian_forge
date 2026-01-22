import unittest
from apitools.base.protorpclite import messages
from apitools.base.py import encoding
from apitools.base.py import exceptions
from apitools.base.py import util
def testReservedExpansion(self):
    method_config_reserved = MockedMethodConfig(relative_path='{+x}/baz', path_params=['x'])
    self.assertEquals('foo/:bar:/baz', util.ExpandRelativePath(method_config_reserved, {'x': 'foo/:bar:'}))
    method_config_no_reserved = MockedMethodConfig(relative_path='{x}/baz', path_params=['x'])
    self.assertEquals('foo%2F%3Abar%3A/baz', util.ExpandRelativePath(method_config_no_reserved, {'x': 'foo/:bar:'}))