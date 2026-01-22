from unittest import mock
import ddt
from oslo_utils import uuidutils
import manilaclient
from manilaclient import exceptions
from manilaclient.tests.unit import utils
from manilaclient.v2 import client
@mock.patch.object(client.ks_client, 'Client', mock.Mock())
@mock.patch.object(client.session.discover, 'Discover', mock.Mock())
@mock.patch.object(client.session, 'Session', mock.Mock())
def test_client_init_no_session_no_auth_token_endpoint_not_found(self):
    self.mock_object(client.httpclient, 'HTTPClient')
    client_args = self._get_client_args(auth_urli='fake_url', password='foo_password', tenant_id='foo_tenant_id')
    discover = client.session.discover.Discover
    discover.return_value.url_for.return_value = None
    mocked_ks_client = client.ks_client.Client.return_value
    self.assertRaises(exceptions.CommandError, client.Client, **client_args)
    self.assertTrue(client.session.Session.called)
    self.assertTrue(client.session.discover.Discover.called)
    self.assertFalse(client.httpclient.HTTPClient.called)
    self.assertFalse(client.ks_client.Client.called)
    self.assertFalse(mocked_ks_client.service_catalog.get_endpoints.called)
    self.assertFalse(mocked_ks_client.authenticate.called)