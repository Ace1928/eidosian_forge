from osc_lib.tests import utils as osc_utils
from unittest import mock
from saharaclient.api import node_group_templates as api_ngt
from saharaclient.osc.v2 import node_group_templates as osc_ngt
from saharaclient.tests.unit.osc.v1 import fakes
def test_ngt_list_long(self):
    arglist = ['--long']
    verifylist = [('long', True)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    expected_columns = ['Name', 'Id', 'Plugin name', 'Plugin version', 'Node processes', 'Description']
    self.assertEqual(expected_columns, columns)
    expected_data = [('template', 'ng_id', 'fake', '0.1', 'namenode, tasktracker', 'description')]
    self.assertEqual(expected_data, list(data))