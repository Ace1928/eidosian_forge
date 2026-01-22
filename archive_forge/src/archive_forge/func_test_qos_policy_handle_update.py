from unittest import mock
from heat.common import template_format
from heat.engine.clients.os import neutron
from heat.engine import rsrc_defn
from heat.engine import stack
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_qos_policy_handle_update(self):
    policy_id = '477e8273-60a7-4c41-b683-fdb0bc7cd151'
    self.my_qos_policy.resource_id = policy_id
    props = {'name': 'test_policy', 'description': 'test', 'shared': False}
    prop_dict = props.copy()
    update_snippet = rsrc_defn.ResourceDefinition(self.my_qos_policy.name, self.my_qos_policy.type(), props)
    self.my_qos_policy.handle_update(json_snippet=update_snippet, tmpl_diff={}, prop_diff=props)
    props['name'] = None
    self.my_qos_policy.handle_update(json_snippet=update_snippet, tmpl_diff={}, prop_diff=props)
    self.assertEqual(2, self.neutronclient.update_qos_policy.call_count)
    self.neutronclient.update_qos_policy.assert_called_with(policy_id, {'policy': prop_dict})