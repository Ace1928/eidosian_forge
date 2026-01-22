from unittest import mock
from magnumclient.common.apiclient import exceptions
from magnumclient.tests.v1 import shell_test_base
from magnumclient.v1.cluster_templates import ClusterTemplate
@mock.patch('magnumclient.v1.cluster_templates.ClusterTemplateManager.create')
def test_cluster_template_create_success_with_master_flavor(self, mock_create):
    self._test_arg_success('cluster-template-create --name test --image-id test_image --keypair-id test_keypair --external-network-id test_net --coe swarm --dns-nameserver test_dns --master-flavor-id test_flavor')
    expected_args = self._get_expected_args(name='test', image_id='test_image', keypair_id='test_keypair', coe='swarm', external_network_id='test_net', dns_nameserver='test_dns', master_flavor_id='test_flavor')
    mock_create.assert_called_with(**expected_args)