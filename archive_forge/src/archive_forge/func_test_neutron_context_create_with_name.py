from unittest import mock
from oslo_config import cfg
from oslo_context import context as oslo_context
from testtools import matchers
from neutron_lib import context
from neutron_lib.tests import _base
def test_neutron_context_create_with_name(self):
    ctx = context.Context('user_id', 'tenant_id', tenant_name='tenant_name', user_name='user_name')
    self.assertEqual('user_name', ctx.user_name)
    self.assertEqual('tenant_name', ctx.tenant_name)
    self.assertEqual('tenant_name', ctx.project_name)
    self.assertEqual('user_id', ctx.user_id)
    self.assertEqual('tenant_id', ctx.tenant_id)