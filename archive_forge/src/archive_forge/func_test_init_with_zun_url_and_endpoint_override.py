from unittest import mock
import testtools
from zunclient.v1 import client
@mock.patch('zunclient.common.httpclient.SessionClient')
@mock.patch('keystoneauth1.session.Session')
def test_init_with_zun_url_and_endpoint_override(self, mock_session, http_client):
    session = mock.Mock()
    client.Client(session=session, zun_url='zunurl', endpoint_override='zunurl')
    mock_session.assert_not_called()
    http_client.assert_called_once_with(interface='public', region_name=None, service_name=None, service_type='container', session=session, endpoint_override='zunurl', api_version=None)