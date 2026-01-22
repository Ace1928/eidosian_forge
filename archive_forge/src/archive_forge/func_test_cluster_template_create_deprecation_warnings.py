from unittest import mock
from magnumclient.common.apiclient import exceptions
from magnumclient.tests.v1 import shell_test_base
from magnumclient.v1.cluster_templates import ClusterTemplate
@mock.patch('magnumclient.v1.cluster_templates.ClusterTemplateManager.create')
def test_cluster_template_create_deprecation_warnings(self, mock_create):
    required_args = 'cluster-template-create --coe test --external-network public --image test '
    self._test_arg_failure('cluster-template-create --coe test --external-network-id test --image test ', self._deprecated_warning)
    expected_args = self._get_expected_args(image_id='test', coe='test', external_network_id='test')
    mock_create.assert_called_with(**expected_args)
    self._test_arg_failure('cluster-template-create --coe test --external-network test --image-id test ', self._deprecated_warning)
    expected_args = self._get_expected_args(image_id='test', coe='test', external_network_id='test')
    mock_create.assert_called_with(**expected_args)
    self._test_arg_failure('cluster-template-create --coe test --external-network-id test --image-id test ', self._deprecated_warning)
    expected_args = self._get_expected_args(image_id='test', coe='test', external_network_id='test')
    mock_create.assert_called_with(**expected_args)
    self._test_arg_failure(required_args + '--keypair-id test', self._deprecated_warning)
    expected_args = self._get_expected_args(image_id='test', coe='test', keypair_id='test', external_network_id='public')
    mock_create.assert_called_with(**expected_args)
    self._test_arg_failure(required_args + '--flavor-id test', self._deprecated_warning)
    expected_args = self._get_expected_args(image_id='test', coe='test', flavor_id='test', external_network_id='public')
    mock_create.assert_called_with(**expected_args)
    self._test_arg_failure(required_args + '--master-flavor-id test', self._deprecated_warning)
    expected_args = self._get_expected_args(image_id='test', coe='test', master_flavor_id='test', external_network_id='public')
    mock_create.assert_called_with(**expected_args)
    self._test_arg_failure(required_args + '--name foo', self._deprecated_warning)
    expected_args = self._get_expected_args(image_id='test', coe='test', name='foo', external_network_id='public')
    mock_create.assert_called_with(**expected_args)