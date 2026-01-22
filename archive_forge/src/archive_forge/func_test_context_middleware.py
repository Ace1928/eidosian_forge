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
def test_context_middleware(self):
    middleware = context.ContextMiddleware(None, None)
    request = webob.Request.blank('/stacks', headers=self.headers, environ=self.environ)
    self.assertIsNone(middleware.process_request(request))
    ctx = request.context.to_dict()
    for k, v in self.context_dict.items():
        self.assertEqual(v, ctx[k], 'Key %s values do not match' % k)
    self.assertIsNotNone(ctx.get('request_id'))