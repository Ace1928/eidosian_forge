from unittest import mock
from oslo_config import cfg
from oslo_context import context as oslo_context
from testtools import matchers
from neutron_lib import context
from neutron_lib.tests import _base
def test_neutron_context_to_dict(self):
    ctx = context.Context('user_id', 'tenant_id')
    ctx_dict = ctx.to_dict()
    self.assertEqual('user_id', ctx_dict['user_id'])
    self.assertEqual('tenant_id', ctx_dict['project_id'])
    self.assertEqual(ctx.request_id, ctx_dict['request_id'])
    self.assertEqual('user_id', ctx_dict['user'])
    self.assertIsNone(ctx_dict['user_name'])
    self.assertIsNone(ctx_dict['tenant_name'])
    self.assertIsNone(ctx_dict['project_name'])
    self.assertIsNone(ctx_dict['auth_token'])