from unittest import mock
from heat.common import template_format
from heat.engine.clients.os import neutron
from heat.engine import rsrc_defn
from heat.engine import stack
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_rule_handle_delete(self):
    rule_id = 'cf0eab12-ef8b-4a62-98d0-70576583c17a'
    self.minimum_packet_rate_rule.resource_id = rule_id
    self.neutronclient.delete_minimum_packet_rate_rule.return_value = None
    self.assertIsNone(self.minimum_packet_rate_rule.handle_delete())
    self.neutronclient.delete_minimum_packet_rate_rule.assert_called_once_with(rule_id, self.policy_id)