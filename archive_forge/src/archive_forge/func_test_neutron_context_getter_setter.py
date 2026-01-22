from unittest import mock
from oslo_config import cfg
from oslo_context import context as oslo_context
from testtools import matchers
from neutron_lib import context
from neutron_lib.tests import _base
def test_neutron_context_getter_setter(self):
    ctx = context.Context('Anakin', 'Skywalker')
    self.assertEqual('Anakin', ctx.user_id)
    self.assertEqual('Skywalker', ctx.tenant_id)
    ctx.user_id = 'Darth'
    ctx.tenant_id = 'Vader'
    self.assertEqual('Darth', ctx.user_id)
    self.assertEqual('Vader', ctx.tenant_id)