import json
from unittest import mock
import requests
from cinderclient import exceptions
from cinderclient.tests.unit import utils
from cinderclient.v3 import client
def test_authenticate_tenant_id(self):
    cs = client.Client('username', 'password', auth_url='http://localhost:8776/v2', tenant_id='tenant_id', service_type='volumev2')
    resp = {'access': {'token': {'expires': '2014-11-01T03:32:15-05:00', 'id': 'FAKE_ID', 'tenant': {'description': None, 'enabled': True, 'id': 'tenant_id', 'name': 'demo'}}, 'serviceCatalog': [{'type': 'volumev2', 'endpoints': [{'region': 'RegionOne', 'adminURL': 'http://localhost:8776/v2', 'internalURL': 'http://localhost:8776/v2', 'publicURL': 'http://localhost:8776/v2'}]}]}}
    auth_response = utils.TestResponse({'status_code': 200, 'text': json.dumps(resp)})
    mock_request = mock.Mock(return_value=auth_response)

    @mock.patch.object(requests, 'request', mock_request)
    def test_auth_call():
        cs.client.authenticate()
        headers = {'User-Agent': cs.client.USER_AGENT, 'Content-Type': 'application/json', 'Accept': 'application/json'}
        body = {'auth': {'passwordCredentials': {'username': cs.client.user, 'password': cs.client.password}, 'tenantId': cs.client.tenant_id}}
        token_url = cs.client.auth_url + '/tokens'
        mock_request.assert_called_with('POST', token_url, headers=headers, data=json.dumps(body), allow_redirects=True, **self.TEST_REQUEST_BASE)
        endpoints = resp['access']['serviceCatalog'][0]['endpoints']
        public_url = endpoints[0]['publicURL'].rstrip('/')
        self.assertEqual(public_url, cs.client.management_url)
        token_id = resp['access']['token']['id']
        self.assertEqual(token_id, cs.client.auth_token)
        tenant_id = resp['access']['token']['tenant']['id']
        self.assertEqual(tenant_id, cs.client.tenant_id)
    test_auth_call()