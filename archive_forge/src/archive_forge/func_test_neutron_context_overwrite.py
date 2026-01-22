from unittest import mock
from oslo_config import cfg
from oslo_context import context as oslo_context
from testtools import matchers
from neutron_lib import context
from neutron_lib.tests import _base
def test_neutron_context_overwrite(self):
    ctx1 = context.Context('user_id', 'tenant_id')
    self.assertEqual(ctx1.request_id, oslo_context.get_current().request_id)
    ctx2 = context.Context('user_id', 'tenant_id')
    self.assertNotEqual(ctx2.request_id, ctx1.request_id)
    self.assertEqual(ctx2.request_id, oslo_context.get_current().request_id)
    ctx3 = context.Context('user_id', 'tenant_id', overwrite=False)
    self.assertNotEqual(ctx3.request_id, ctx2.request_id)
    self.assertEqual(ctx2.request_id, oslo_context.get_current().request_id)