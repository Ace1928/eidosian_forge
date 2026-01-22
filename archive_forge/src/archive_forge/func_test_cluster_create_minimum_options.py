from osc_lib.tests import utils as osc_utils
from unittest import mock
from saharaclient.api import cluster_templates as api_ct
from saharaclient.api import clusters as api_cl
from saharaclient.api import images as api_img
from saharaclient.api import node_group_templates as api_ngt
from saharaclient.osc.v2 import clusters as osc_cl
from saharaclient.tests.unit.osc.v1 import test_clusters as tc_v1
def test_cluster_create_minimum_options(self):
    arglist = ['--name', 'fake', '--cluster-template', 'template', '--image', 'ubuntu']
    verifylist = [('name', 'fake'), ('cluster_template', 'template'), ('image', 'ubuntu')]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    self.cl_mock.create.assert_called_once_with(cluster_template_id='ct_id', count=None, default_image_id='img_id', description=None, plugin_version='0.1', is_protected=False, is_public=False, is_transient=False, name='fake', net_id=None, plugin_name='fake', user_keypair_id=None)