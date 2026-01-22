from unittest import mock
from neutronclient.common import exceptions
from neutronclient.v2_0 import client as neutronclient
from oslo_config import cfg
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import neutron
from heat.engine.resources.openstack.neutron import firewall
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
def test_update_with_value_specs(self):
    rsrc = self.create_firewall(value_specs=False)
    self.mockclient.show_firewall.return_value = {'firewall': {'status': 'ACTIVE'}}
    scheduler.TaskRunner(rsrc.create)()
    prop_diff = {'value_specs': {'router_ids': ['router_1', 'router_2']}}
    update_snippet = rsrc_defn.ResourceDefinition(rsrc.name, rsrc.type(), prop_diff)
    rsrc.handle_update(update_snippet, {}, prop_diff)
    self.mockclient.create_firewall.assert_called_once_with({'firewall': {'name': 'test-firewall', 'admin_state_up': True, 'firewall_policy_id': 'policy-id', 'shared': True}})
    self.mockclient.show_firewall.assert_called_once_with('5678')
    self.mockclient.update_firewall.assert_called_once_with('5678', {'firewall': {'router_ids': ['router_1', 'router_2']}})