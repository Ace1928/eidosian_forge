from unittest import mock
from heat.common import template_format
from heat.engine.clients.os import neutron
from heat.engine import rsrc_defn
from heat.engine import stack
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_qos_policy_handle_delete(self):
    policy_id = '477e8273-60a7-4c41-b683-fdb0bc7cd151'
    self.my_qos_policy.resource_id = policy_id
    self.neutronclient.delete_qos_policy.return_value = None
    self.assertIsNone(self.my_qos_policy.handle_delete())
    self.neutronclient.delete_qos_policy.assert_called_once_with(self.my_qos_policy.resource_id)