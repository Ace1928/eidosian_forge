import json
from unittest import mock
import uuid
from keystoneauth1 import access as ks_access
from keystoneauth1 import exceptions as kc_exception
from keystoneauth1.identity import access as ks_auth_access
from keystoneauth1.identity import generic as ks_auth
from keystoneauth1 import loading as ks_loading
from keystoneauth1 import session as ks_session
from keystoneauth1 import token_endpoint as ks_token_endpoint
from keystoneclient.v3 import client as kc_v3
from keystoneclient.v3 import domains as kc_v3_domains
from oslo_config import cfg
from heat.common import config
from heat.common import exception
from heat.common import password_gen
from heat.engine.clients.os.keystone import heat_keystoneclient
from heat.tests import common
from heat.tests import utils
def test_get_ec2_keypair_id(self):
    """Test getting ec2 credential by id."""
    user_id = 'atestuser'
    self._stubs_auth(user_id=user_id)
    ctx = utils.dummy_context()
    ctx.trust_id = None
    ex_data = {'access': 'access123', 'secret': 'secret456'}
    ex_data_json = json.dumps(ex_data)
    credential_id = 'acredential123'
    mock_cred = mock.Mock()
    mock_cred.id = credential_id
    mock_cred.user_id = user_id
    mock_cred.blob = ex_data_json
    mock_cred.type = 'ec2'
    self.mock_ks_v3_client.credentials.get.return_value = mock_cred
    heat_ks_client = heat_keystoneclient.KeystoneClient(ctx)
    ec2_cred = heat_ks_client.get_ec2_keypair(credential_id=credential_id)
    self.assertEqual(credential_id, ec2_cred.id)
    self.assertEqual('access123', ec2_cred.access)
    self.assertEqual('secret456', ec2_cred.secret)
    self.mock_ks_v3_client.credentials.get.assert_called_once_with(credential_id)
    self._validate_stub_auth()