from unittest import mock
from heat.common import template_format
from heat.engine.clients.os import neutron
from heat.engine import rsrc_defn
from heat.engine import stack
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_rule_handle_create(self):
    rule = {'minimum_packet_rate_rule': {'id': 'cf0eab12-ef8b-4a62-98d0-70576583c17a', 'min_kpps': 1000, 'direction': 'any', 'tenant_id': 'd66c74c01d6c41b9846088c1ad9634d0'}}
    create_props = {'min_kpps': 1000, 'direction': 'any', 'tenant_id': 'd66c74c01d6c41b9846088c1ad9634d0'}
    self.neutronclient.create_minimum_packet_rate_rule.return_value = rule
    self.minimum_packet_rate_rule.handle_create()
    self.assertEqual('cf0eab12-ef8b-4a62-98d0-70576583c17a', self.minimum_packet_rate_rule.resource_id)
    self.neutronclient.create_minimum_packet_rate_rule.assert_called_once_with(self.policy_id, {'minimum_packet_rate_rule': create_props})