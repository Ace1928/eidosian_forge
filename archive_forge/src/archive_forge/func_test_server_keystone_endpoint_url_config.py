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
def test_server_keystone_endpoint_url_config(self):
    """Return non fallback url path."""
    cfg.CONF.set_override('server_keystone_endpoint_type', 'public')
    ctx = utils.dummy_context()
    ctx.trust_id = None
    heat_ks_client = heat_keystoneclient.KeystoneClient(ctx)
    fallback_url = 'http://server.fallback.test:5000/v3'
    auth_ref = heat_ks_client.context.auth_plugin.get_access(heat_ks_client.session)
    auth_ref.service_catalog.get_urls = mock.MagicMock()
    auth_ref.service_catalog.get_urls.return_value = ['http://server.public.test:5000']
    self.assertEqual(heat_ks_client.server_keystone_endpoint_url(fallback_url), 'http://server.public.test:5000/v3')
    cfg.CONF.clear_override('server_keystone_endpoint_type')