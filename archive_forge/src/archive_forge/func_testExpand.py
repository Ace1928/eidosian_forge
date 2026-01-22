import unittest
from apitools.base.protorpclite import messages
from apitools.base.py import encoding
from apitools.base.py import exceptions
from apitools.base.py import util
def testExpand(self):
    method_config_xy = MockedMethodConfig(relative_path='{x}/y/{z}', path_params=['x', 'z'])
    self.assertEquals(util.ExpandRelativePath(method_config_xy, {'x': '1', 'z': '2'}), '1/y/2')
    self.assertEquals(util.ExpandRelativePath(method_config_xy, {'x': '1', 'z': '2'}, relative_path='{x}/y/{z}/q'), '1/y/2/q')