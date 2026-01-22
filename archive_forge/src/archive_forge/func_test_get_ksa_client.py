from unittest import mock
from oslo_config import cfg
from glance import context
from glance.tests.unit import utils as unit_utils
from glance.tests import utils
@mock.patch('keystoneauth1.token_endpoint.Token')
@mock.patch('keystoneauth1.session.Session')
def test_get_ksa_client(self, mock_session, mock_token):
    ctx = context.RequestContext(auth_token='token')
    CONF.register_group(cfg.OptGroup('keystone_authtoken'))
    with mock.patch.object(CONF, 'keystone_authtoken') as ksat:
        ksat.identity_uri = 'http://keystone'
        client = context.get_ksa_client(ctx)
    self.assertEqual(mock_session.return_value, client)
    mock_session.assert_called_once_with(auth=mock_token.return_value)
    mock_token.assert_called_once_with('http://keystone', 'token')