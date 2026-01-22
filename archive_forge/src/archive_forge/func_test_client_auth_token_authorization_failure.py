from unittest import mock
import fixtures
from keystoneauth1 import adapter
import logging
import requests
import testtools
from troveclient.apiclient import client
from troveclient import client as other_client
from troveclient import exceptions
from troveclient import service_catalog
import troveclient.v1.client
@mock.patch.object(service_catalog, 'ServiceCatalog', side_effect=KeyError)
def test_client_auth_token_authorization_failure(self, mock_service_catalog):
    auth_url = 'http://www.blah.com'
    proxy_token = 'foobar'
    proxy_tenant_id = 'user'
    mock_service_catalog.return_value.get_token = mock.Mock(return_value=proxy_token)
    instance = other_client.HTTPClient(proxy_token=proxy_token, proxy_tenant_id=proxy_tenant_id, user=None, password=None, tenant_id=proxy_tenant_id, projectid=None, timeout=2, auth_url=auth_url)
    instance.management_url = 'http://example.com'
    instance.get_service_url = mock.Mock(return_value='http://example.com')
    instance.version = 'v2.0'
    mock_request = mock.Mock()
    mock_request.return_value = requests.Response()
    mock_request.return_value.status_code = 200
    mock_request.return_value.headers = {'x-server-management-url': 'blah.com', 'x-auth-token': 'blah'}
    with mock.patch('requests.request', mock_request):
        self.assertRaises(exceptions.AuthorizationFailure, instance.authenticate)