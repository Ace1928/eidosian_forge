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
def test_regenerate_trust_context_with_no_exist_trust_id(self):
    """Test regenerate_trust_context."""

    class MockTrust(object):
        id = 'dtrust123'
    mock_ks_auth, mock_auth_ref = self._stubs_auth(user_id='5678', project_id='42', stub_trust_context=True, stub_admin_auth=True)
    cfg.CONF.set_override('deferred_auth_method', 'trusts')
    trustor_roles = ['heat_stack_owner', 'admin', '__member__']
    trustee_roles = trustor_roles
    mock_auth_ref.user_id = '5678'
    mock_auth_ref.project_id = '42'
    self.mock_ks_v3_client.trusts.create.return_value = MockTrust()
    ctx = utils.dummy_context(roles=trustor_roles)
    ctx.trust_id = None
    heat_ks_client = heat_keystoneclient.KeystoneClient(ctx)
    trust_context = heat_ks_client.regenerate_trust_context()
    self.assertEqual('dtrust123', trust_context.trust_id)
    self.assertEqual('5678', trust_context.trustor_user_id)
    ks_loading.load_auth_from_conf_options.assert_called_once_with(cfg.CONF, 'trustee', trust_id=None)
    self.mock_ks_v3_client.trusts.create.assert_called_once_with(trustor_user='5678', trustee_user='1234', project='42', impersonation=True, allow_redelegation=False, role_names=trustee_roles)
    self.assertEqual(0, self.mock_ks_v3_client.trusts.delete.call_count)