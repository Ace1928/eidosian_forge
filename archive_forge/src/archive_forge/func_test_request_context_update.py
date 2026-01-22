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
def test_request_context_update(self):
    ctx = context.RequestContext.from_dict(self.ctx)
    for k in self.ctx:
        if k == 'user_identity' or k == 'user_domain_id' or k == 'project_domain_id':
            continue
        if k == 'tenant' or k == 'user':
            continue
        self.assertEqual(self.ctx.get(k), ctx.to_dict().get(k))
        override = '%s_override' % k
        setattr(ctx, k, override)
        self.assertEqual(override, ctx.to_dict().get(k))