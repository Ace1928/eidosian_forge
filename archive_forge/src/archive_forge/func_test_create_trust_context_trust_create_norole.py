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
def test_create_trust_context_trust_create_norole(self):
    """Test create_trust_context when creating a trust."""
    mock_auth, mock_auth_ref = self._stubs_auth(user_id='5678', project_id='42', stub_trust_context=True, stub_admin_auth=True)
    cfg.CONF.set_override('deferred_auth_method', 'trusts')
    cfg.CONF.set_override('trusts_delegated_roles', ['heat_stack_owner'])
    exc = kc_exception.NotFound
    self.mock_ks_v3_client.trusts.create.side_effect = exc
    ctx = utils.dummy_context()
    ctx.trust_id = None
    heat_ks_client = heat_keystoneclient.KeystoneClient(ctx)
    exc = self.assertRaises(exception.MissingCredentialError, heat_ks_client.create_trust_context)
    expected = 'Missing required credential: roles '
    "{'role_names': ['heat_stack_owner']}"
    self.assertIn(expected, str(exc))
    self.m_load_auth.assert_called_with(cfg.CONF, 'trustee', trust_id=None)
    self.mock_ks_v3_client.trusts.create.assert_called_once_with(allow_redelegation=False, trustor_user='5678', trustee_user='1234', project='42', impersonation=True, role_names=['heat_stack_owner'])