import os
import tempfile
from unittest import mock
from oslo_serialization import jsonutils
from oslo_utils import uuidutils
from oslotest import base
import osprofiler.profiler
from mistralclient.api import client
@mock.patch('mistralclient.auth.keystone.KeystoneAuthHandler._is_service_catalog_v2', return_value=True)
@mock.patch('keystoneauth1.identity.generic.Password')
@mock.patch('keystoneauth1.session.Session')
@mock.patch('mistralclient.api.httpclient.HTTPClient')
def test_target_parameters_processed(self, http_client_mock, session_mock, password_mock, catalog_type_mock):
    session = mock.MagicMock()
    target_session = mock.MagicMock()
    session_mock.side_effect = [session, target_session]
    auth = mock.MagicMock()
    target_auth = mock.MagicMock()
    target_auth._plugin.auth_url = AUTH_HTTP_URL_v3
    password_mock.side_effect = [auth, target_auth]
    get_endpoint = mock.Mock(return_value='http://mistral_host:8989/v2')
    session.get_endpoint = get_endpoint
    target_session.get_project_id = mock.Mock(return_value='projectid')
    target_session.get_user_id = mock.Mock(return_value='userid')
    target_session.get_auth_headers = mock.Mock(return_value={'X-Auth-Token': 'authtoken'})
    mock_access = mock.MagicMock()
    mock_catalog = mock.MagicMock()
    mock_catalog.catalog = {}
    mock_access.service_catalog = mock_catalog
    auth.get_access = mock.Mock(return_value=mock_access)
    client.client(auth_url=AUTH_HTTP_URL_v3, username='user', api_key='password', user_domain_name='Default', project_domain_name='Default', target_username='tmistral', target_project_name='tmistralp', target_auth_url=AUTH_HTTP_URL_v3, target_api_key='tpassword', target_user_domain_name='Default', target_project_domain_name='Default', target_region_name='tregion')
    self.assertTrue(http_client_mock.called)
    mistral_url_for_http = http_client_mock.call_args[0][0]
    kwargs = http_client_mock.call_args[1]
    self.assertEqual('http://mistral_host:8989/v2', mistral_url_for_http)
    expected_values = {'target_project_id': 'projectid', 'target_auth_token': 'authtoken', 'target_user_id': 'userid', 'target_auth_url': AUTH_HTTP_URL_v3, 'target_project_name': 'tmistralp', 'target_username': 'tmistral', 'target_region_name': 'tregion', 'target_service_catalog': '{}'}
    for key in expected_values:
        self.assertEqual(expected_values[key], kwargs[key])