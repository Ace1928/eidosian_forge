from osc_lib.tests import utils as osc_utils
from saharaclient.api import cluster_templates as api_ct
from saharaclient.api import node_group_templates as api_ngt
from saharaclient.osc.v2 import cluster_templates as osc_ct
from saharaclient.tests.unit.osc.v1 import test_cluster_templates as tct_v1
def test_ct_list_long(self):
    arglist = ['--long']
    verifylist = [('long', True)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    expected_columns = ['Name', 'Id', 'Plugin name', 'Plugin version', 'Node groups', 'Description']
    self.assertEqual(expected_columns, columns)
    expected_data = [('template', '0647061f-ab98-4c89-84e0-30738ea55750', 'fake', '0.1', 'fakeng:2', 'Cluster template for tests')]
    self.assertEqual(expected_data, list(data))