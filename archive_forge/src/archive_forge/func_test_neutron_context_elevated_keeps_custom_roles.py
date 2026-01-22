from unittest import mock
from oslo_config import cfg
from oslo_context import context as oslo_context
from testtools import matchers
from neutron_lib import context
from neutron_lib.tests import _base
def test_neutron_context_elevated_keeps_custom_roles(self):
    expected_admin_roles = ['admin', 'member', 'reader']
    custom_roles = ['custom_role']
    ctx = context.Context('user_id', 'tenant_id', roles=custom_roles)
    self.assertFalse(ctx.is_admin)
    self.assertNotEqual('all', ctx.system_scope)
    for expected_admin_role in expected_admin_roles:
        self.assertNotIn(expected_admin_role, ctx.roles)
    for custom_role in custom_roles:
        self.assertIn(custom_role, ctx.roles)
    elevated_ctx = ctx.elevated()
    self.assertTrue(elevated_ctx.is_admin)
    for expected_admin_role in expected_admin_roles:
        self.assertIn(expected_admin_role, elevated_ctx.roles)
    for custom_role in custom_roles:
        self.assertIn(custom_role, ctx.roles)