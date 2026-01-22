from unittest import mock
from magnumclient.common.apiclient import exceptions
from magnumclient.tests.v1 import shell_test_base
from magnumclient.v1.cluster_templates import ClusterTemplate
@mock.patch('magnumclient.v1.cluster_templates.ClusterTemplateManager.create')
def test_cluster_template_create_success(self, mock_create):
    self._test_arg_success('cluster-template-create --name test --image-id test_image --keypair-id test_keypair --external-network-id test_net --coe swarm --dns-nameserver test_dns --flavor-id test_flavor --fixed-network private --fixed-subnet private-subnet --volume-driver test_volume --network-driver test_driver --labels key=val --master-flavor-id test_flavor --docker-volume-size 10 --docker-storage-driver devicemapper --public --server-type vm --master-lb-enabled ')
    expected_args = self._get_expected_args(name='test', image_id='test_image', keypair_id='test_keypair', coe='swarm', external_network_id='test_net', dns_nameserver='test_dns', public=True, flavor_id='test_flavor', master_flavor_id='test_flavor', fixed_network='private', fixed_subnet='private-subnet', server_type='vm', network_driver='test_driver', volume_driver='test_volume', docker_storage_driver='devicemapper', docker_volume_size=10, master_lb_enabled=True, labels={'key': 'val'})
    mock_create.assert_called_with(**expected_args)
    self._test_arg_success('cluster-template-create --keypair-id test_keypair --external-network-id test_net --image-id test_image --coe kubernetes --name test --server-type vm')
    expected_args = self._get_expected_args(name='test', image_id='test_image', keypair_id='test_keypair', coe='kubernetes', external_network_id='test_net', server_type='vm')
    mock_create.assert_called_with(**expected_args)