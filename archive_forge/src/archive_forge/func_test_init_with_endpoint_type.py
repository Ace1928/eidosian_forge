import testtools
from unittest import mock
from keystoneauth1.exceptions import catalog
from magnumclient.v1 import client
@mock.patch('magnumclient.common.httpclient.SessionClient')
@mock.patch('magnumclient.v1.client._load_session')
@mock.patch('magnumclient.v1.client._load_service_type', return_value='container-infra')
def test_init_with_endpoint_type(self, mock_load_service_type, mock_load_session, mock_http_client):
    self._test_init_with_interface(lambda x: client.Client(interface='public', endpoint_type='%sURL' % x), mock_load_service_type, mock_load_session, mock_http_client)