import copy
from unittest import mock
from keystoneauth1 import exceptions as ksa_exceptions
from keystoneauth1 import session as ksa_session
from openstack.config import cloud_region
from os_client_config import cloud_config
from os_client_config import defaults
from os_client_config import exceptions
from os_client_config.tests import base
@mock.patch.object(cloud_region.CloudRegion, 'get_auth_args')
def test_legacy_client_object_store_endpoint(self, mock_get_auth_args):
    mock_client = mock.Mock()
    mock_get_auth_args.return_value = {}
    config_dict = defaults.get_defaults()
    config_dict.update(fake_services_dict)
    config_dict['object_store_endpoint'] = 'http://example.com/swift'
    cc = cloud_config.CloudConfig('test1', 'region-al', config_dict, auth_plugin=mock.Mock())
    cc.get_legacy_client('object-store', mock_client)
    mock_client.assert_called_with(session=mock.ANY, os_options={'region_name': 'region-al', 'service_type': 'object-store', 'object_storage_url': 'http://example.com/swift', 'endpoint_type': 'public'})