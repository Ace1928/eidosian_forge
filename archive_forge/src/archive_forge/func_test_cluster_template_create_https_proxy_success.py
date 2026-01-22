from unittest import mock
from magnumclient.common.apiclient import exceptions
from magnumclient.tests.v1 import shell_test_base
from magnumclient.v1.cluster_templates import ClusterTemplate
@mock.patch('magnumclient.v1.cluster_templates.ClusterTemplateManager.create')
def test_cluster_template_create_https_proxy_success(self, mock_create):
    self._test_arg_success('cluster-template-create --name test --fixed-network private --keypair-id test_keypair --external-network-id test_net --image-id test_image --coe swarm --https-proxy https_proxy --server-type vm')
    expected_args = self._get_expected_args(name='test', image_id='test_image', keypair_id='test_keypair', coe='swarm', external_network_id='test_net', fixed_network='private', server_type='vm', https_proxy='https_proxy')
    mock_create.assert_called_with(**expected_args)