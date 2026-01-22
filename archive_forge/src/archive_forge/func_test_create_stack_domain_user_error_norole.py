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
def test_create_stack_domain_user_error_norole(self):
    """Test creating a stack domain user, no role error path."""
    ctx = utils.dummy_context()
    self.patchobject(ctx, '_create_auth_plugin')
    ctx.trust_id = None
    self._stub_domain_admin_client(domain_id=None)
    self.mock_ks_v3_client.roles.list.return_value = []
    heat_ks_client = heat_keystoneclient.KeystoneClient(ctx)
    err = self.assertRaises(exception.Error, heat_ks_client.create_stack_domain_user, username='duser', project_id='aproject')
    self.assertIn("Can't find role heat_stack_user", str(err))
    self._validate_stub_domain_admin_client()
    self.mock_ks_v3_client.roles.list.assert_called_once_with(name='heat_stack_user')