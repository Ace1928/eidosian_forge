from unittest import mock
from oslo_config import cfg
from oslo_context import context as oslo_context
from testtools import matchers
from neutron_lib import context
from neutron_lib.tests import _base
def test_neutron_context_admin_without_session_to_dict(self):
    ctx = context.get_admin_context_without_session()
    ctx_dict = ctx.to_dict()
    self.assertIsNone(ctx_dict['user_id'])
    self.assertIsNone(ctx_dict['tenant_id'])
    self.assertIsNone(ctx_dict['auth_token'])
    self.assertIn('admin', ctx_dict['roles'])
    self.assertFalse(hasattr(ctx, 'session'))