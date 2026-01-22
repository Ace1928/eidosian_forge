import os
from unittest import mock
from keystoneauth1 import loading as ks_loading
from oslo_config import cfg
from oslo_config import fixture as config_fixture
from oslo_middleware import request_id
from oslo_policy import opts as policy_opts
from oslo_utils import importutils
import webob
from heat.common import context
from heat.common import exception
from heat.tests import common
def test_keystone_v3_endpoint_in_keystone_authtoken_config(self):
    """Ensure that the [keystone_authtoken] section is used.

        Ensure that the [keystone_authtoken] section of the configuration
        is used when the auth_uri is not defined in the context or the
        [clients_keystone] section.
        """
    importutils.import_module('keystonemiddleware.auth_token')
    try:
        cfg.CONF.set_override('www_authenticate_uri', 'http://abc/v2.0', group='keystone_authtoken')
    except cfg.NoSuchOptError:
        cfg.CONF.set_override('auth_uri', 'http://abc/v2.0', group='keystone_authtoken')
    policy_check = 'heat.common.policy.Enforcer.check_is_admin'
    with mock.patch(policy_check) as pc:
        pc.return_value = False
        ctx = context.RequestContext(auth_url=None)
        self.assertEqual(ctx.keystone_v3_endpoint, 'http://abc/v3')