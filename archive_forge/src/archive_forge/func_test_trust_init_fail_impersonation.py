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
def test_trust_init_fail_impersonation(self):
    """Test consuming a trust when initializing, impersonation error."""
    self._stubs_auth(method='trust', user_id='wrong_user_id')
    cfg.CONF.set_override('deferred_auth_method', 'trusts')
    ctx = utils.dummy_context()
    ctx.username = 'heat'
    ctx.password = None
    ctx.auth_token = None
    ctx.trust_id = 'atrust123'
    ctx.trustor_user_id = 'trustor_user_id'
    self.assertRaises(exception.AuthorizationFailure, heat_keystoneclient.KeystoneClient, ctx)
    self._validate_stub_auth()