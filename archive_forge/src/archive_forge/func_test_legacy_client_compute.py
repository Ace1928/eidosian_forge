import copy
from unittest import mock
from keystoneauth1 import exceptions as ksa_exceptions
from keystoneauth1 import session as ksa_session
from openstack.config import cloud_region
from os_client_config import cloud_config
from os_client_config import defaults
from os_client_config import exceptions
from os_client_config.tests import base
@mock.patch.object(cloud_region.CloudRegion, 'get_session_endpoint')
def test_legacy_client_compute(self, mock_get_session_endpoint):
    mock_client = mock.Mock()
    mock_get_session_endpoint.return_value = 'http://example.com/v2'
    config_dict = defaults.get_defaults()
    config_dict.update(fake_services_dict)
    cc = cloud_config.CloudConfig('test1', 'region-al', config_dict, auth_plugin=mock.Mock())
    cc.get_legacy_client('compute', mock_client)
    mock_client.assert_called_with(version='2', endpoint_type='public', endpoint_override='http://compute.example.com', region_name='region-al', service_type='compute', session=mock.ANY, service_name=None)