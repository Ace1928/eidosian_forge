from osc_lib.tests import utils as osc_utils
from saharaclient.api import cluster_templates as api_ct
from saharaclient.api import node_group_templates as api_ngt
from saharaclient.osc.v2 import cluster_templates as osc_ct
from saharaclient.tests.unit.osc.v1 import test_cluster_templates as tct_v1
def test_ct_update_nothing_updated(self):
    arglist = ['template']
    verifylist = [('cluster_template', 'template')]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    self.ct_mock.update.assert_called_once_with('0647061f-ab98-4c89-84e0-30738ea55750')