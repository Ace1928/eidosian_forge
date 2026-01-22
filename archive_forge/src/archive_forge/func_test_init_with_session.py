import testtools
from unittest import mock
from keystoneauth1.exceptions import catalog
from magnumclient.v1 import client
@mock.patch('magnumclient.common.httpclient.SessionClient')
@mock.patch('magnumclient.v1.client._load_session')
@mock.patch('magnumclient.v1.client._load_service_type', return_value='container-infra')
def test_init_with_session(self, mock_load_service_type, mock_load_session, mock_http_client):
    session = mock.Mock()
    client.Client(session=session)
    mock_load_session.assert_not_called()
    mock_load_service_type.assert_called_once_with(session, **self._load_service_type_kwargs())
    mock_http_client.assert_called_once_with(**self._session_client_kwargs(session))