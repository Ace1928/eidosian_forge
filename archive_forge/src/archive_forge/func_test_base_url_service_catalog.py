import calendar
from unittest import mock
from barbicanclient import exceptions as barbican_exceptions
from keystoneauth1 import identity
from keystoneauth1 import service_token
from oslo_context import context
from oslo_utils import timeutils
from oslo_utils import uuidutils
from castellan.common import exception
from castellan.common.objects import symmetric_key as sym_key
from castellan.key_manager import barbican_key_manager
from castellan.tests.unit.key_manager import test_key_manager
def test_base_url_service_catalog(self):
    endpoint_data = mock.Mock()
    endpoint_data.api_version = 'v321'
    auth = mock.Mock(spec=['service_catalog'])
    auth.service_catalog.endpoint_data_for.return_value = endpoint_data
    endpoint = 'http://localhost/key_manager'
    base_url = self.key_mgr._create_base_url(auth, mock.Mock(), endpoint)
    self.assertEqual(endpoint + '/' + endpoint_data.api_version, base_url)
    auth.service_catalog.endpoint_data_for.assert_called_once_with(service_type='key-manager', interface='public', region_name=None)