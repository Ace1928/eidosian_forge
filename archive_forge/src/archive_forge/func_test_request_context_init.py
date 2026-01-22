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
def test_request_context_init(self):
    ctx = context.RequestContext(auth_token=self.ctx.get('auth_token'), username=self.ctx.get('username'), password=self.ctx.get('password'), aws_creds=self.ctx.get('aws_creds'), project_name=self.ctx.get('tenant'), project_id=self.ctx.get('tenant_id'), user=self.ctx.get('user_id'), auth_url=self.ctx.get('auth_url'), roles=self.ctx.get('roles'), show_deleted=self.ctx.get('show_deleted'), is_admin=self.ctx.get('is_admin'), auth_token_info=self.ctx.get('auth_token_info'), trustor_user_id=self.ctx.get('trustor_user_id'), trust_id=self.ctx.get('trust_id'), region_name=self.ctx.get('region_name'), user_domain_id=self.ctx.get('user_domain_id'), project_domain_id=self.ctx.get('project_domain_id'))
    ctx_dict = ctx.to_dict()
    del ctx_dict['request_id']
    del ctx_dict['project_id']
    del ctx_dict['project_name']
    self.assertEqual(self.ctx, ctx_dict)