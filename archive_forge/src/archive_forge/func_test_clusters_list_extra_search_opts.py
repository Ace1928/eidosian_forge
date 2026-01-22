from osc_lib.tests import utils as osc_utils
from unittest import mock
from saharaclient.api import cluster_templates as api_ct
from saharaclient.api import clusters as api_cl
from saharaclient.api import images as api_img
from saharaclient.api import node_group_templates as api_ngt
from saharaclient.osc.v2 import clusters as osc_cl
from saharaclient.tests.unit.osc.v1 import test_clusters as tc_v1
def test_clusters_list_extra_search_opts(self):
    arglist = ['--plugin', 'fake', '--plugin-version', '0.1', '--name', 'fake']
    verifylist = [('plugin', 'fake'), ('plugin_version', '0.1'), ('name', 'fake')]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    expected_columns = ['Name', 'Id', 'Plugin name', 'Plugin version', 'Status']
    self.assertEqual(expected_columns, columns)
    expected_data = [('fake', 'cluster_id', 'fake', '0.1', 'Active')]
    self.assertEqual(expected_data, list(data))