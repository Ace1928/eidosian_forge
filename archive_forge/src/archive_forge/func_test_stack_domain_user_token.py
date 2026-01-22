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
def test_stack_domain_user_token(self):
    """Test stack_domain_user_token function."""
    dum_tok = 'dummytoken'
    ctx = utils.dummy_context()
    mock_ks_auth = mock.Mock()
    mock_ks_auth.get_token.return_value = dum_tok
    self.patchobject(ctx, '_create_auth_plugin')
    ks_auth.Password.return_value = mock_ks_auth
    ctx.trust_id = None
    heat_ks_client = heat_keystoneclient.KeystoneClient(ctx)
    token = heat_ks_client.stack_domain_user_token(user_id='duser', project_id='aproject', password='apassw')
    self.assertEqual(dum_tok, token)
    ks_auth.Password.assert_called_once_with(auth_url='http://server.test:5000/v3', password='apassw', project_id='aproject', user_id='duser')
    mock_ks_auth.get_token.assert_called_once_with(utils.AnyInstance(ks_session.Session))