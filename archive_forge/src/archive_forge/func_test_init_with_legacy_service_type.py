import testtools
from unittest import mock
from keystoneauth1.exceptions import catalog
from magnumclient.v1 import client
@mock.patch('magnumclient.common.httpclient.SessionClient')
@mock.patch('magnumclient.v1.client._load_session')
def test_init_with_legacy_service_type(self, mock_load_session, mock_http_client):
    session = mock.Mock()
    mock_load_session.return_value = session
    session.get_endpoint.side_effect = [catalog.EndpointNotFound(), mock.Mock()]
    client.Client(username='myuser', auth_url='authurl')
    expected_kwargs = self._session_client_kwargs(session)
    expected_kwargs['service_type'] = 'container'
    mock_http_client.assert_called_once_with(**expected_kwargs)