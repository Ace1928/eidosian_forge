from unittest import mock
from openstackclient.common import module as osc_module
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit import utils
def test_module_list_no_options(self):
    arglist = []
    verifylist = [('all', False)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.assertIn(module_name_1, columns)
    self.assertIn(module_version_1, data)
    self.assertNotIn(module_name_2, columns)
    self.assertNotIn(module_version_2, data)
    self.assertIn(module_name_3, columns)
    self.assertIn(module_version_3, data)
    self.assertNotIn(module_name_4, columns)
    self.assertNotIn(module_version_4, data)
    self.assertNotIn(module_name_5, columns)
    self.assertNotIn(module_version_5, data)