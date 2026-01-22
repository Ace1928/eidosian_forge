from unittest import mock
from heat.common import template_format
from heat.engine.clients.os import neutron
from heat.engine import rsrc_defn
from heat.engine import stack
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_rule_handle_update(self):
    rule_id = 'cf0eab12-ef8b-4a62-98d0-70576583c17a'
    self.minimum_packet_rate_rule.resource_id = rule_id
    prop_diff = {'min_kpps': 500}
    self.minimum_packet_rate_rule.handle_update(json_snippet={}, tmpl_diff={}, prop_diff=prop_diff.copy())
    self.neutronclient.update_minimum_packet_rate_rule.assert_called_once_with(rule_id, self.policy_id, {'minimum_packet_rate_rule': prop_diff})