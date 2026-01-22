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
def test_get_trust_context_auth_plugin_unauthorized(self):
    self.ctx['trust_id'] = 'trust_id'
    ctx = context.RequestContext.from_dict(self.ctx)
    self.patchobject(ks_loading, 'load_auth_from_conf_options', return_value=None)
    self.assertRaises(exception.AuthorizationFailure, getattr, ctx, 'auth_plugin')