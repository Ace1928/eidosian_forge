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
def test_context_middleware_with_requestid(self):
    middleware = context.ContextMiddleware(None, None)
    request = webob.Request.blank('/stacks', headers=self.headers, environ=self.environ)
    req_id = 'req-5a63f0d7-1b69-447b-b621-4ea87cc7186d'
    request.environ[request_id.ENV_REQUEST_ID] = req_id
    self.assertIsNone(middleware.process_request(request))
    ctx = request.context.to_dict()
    for k, v in self.context_dict.items():
        self.assertEqual(v, ctx[k], 'Key %s values do not match' % k)
    self.assertEqual(ctx.get('request_id'), req_id, 'Key request_id values do not match')