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
def test_create_trust_context_trust_create_delegate_all_roleids(self):
    """Test create_trust_context when creating a trust using role IDs."""

    class MockTrust(object):
        id = 'atrust123'
    self._stubs_auth(user_id='5678', project_id='42', stub_trust_context=True, stub_admin_auth=True)
    cfg.CONF.set_override('deferred_auth_method', 'trusts')
    self.mock_ks_v3_client.trusts.create.return_value = MockTrust()
    trustor_roles = [{'name': 'spam', 'id': 'ham'}]
    ctx = utils.dummy_context(roles=trustor_roles)
    ctx.trust_id = None
    heat_ks_client = heat_keystoneclient.KeystoneClient(ctx)
    ctx.auth_token_info = {'token': {'roles': trustor_roles}}
    trust_context = heat_ks_client.create_trust_context()
    self.assertEqual('atrust123', trust_context.trust_id)
    self.assertEqual('5678', trust_context.trustor_user_id)
    args, kwargs = self.mock_ks_v3_client.trusts.create.call_args
    self.assertEqual(['ham'], kwargs['role_ids'])