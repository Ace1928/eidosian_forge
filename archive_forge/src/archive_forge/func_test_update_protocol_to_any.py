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
def test_update_protocol_to_any(self):
    rsrc = self.create_firewall_rule()
    self.mockclient.update_firewall_rule.return_value = None
    scheduler.TaskRunner(rsrc.create)()
    props = self.tmpl['resources']['firewall_rule']['properties'].copy()
    props['protocol'] = 'any'
    update_template = rsrc.t.freeze(properties=props)
    scheduler.TaskRunner(rsrc.update, update_template)()
    self.mockclient.create_firewall_rule.assert_called_once_with({'firewall_rule': {'name': 'test-firewall-rule', 'shared': True, 'action': 'allow', 'protocol': 'tcp', 'enabled': True, 'ip_version': '4'}})
    self.mockclient.update_firewall_rule.assert_called_once_with('5678', {'firewall_rule': {'protocol': None}})