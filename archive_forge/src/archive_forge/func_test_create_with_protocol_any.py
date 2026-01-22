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
def test_create_with_protocol_any(self):
    self.mockclient.create_firewall_rule.return_value = {'firewall_rule': {'id': '5678'}}
    snippet = template_format.parse(firewall_rule_template)
    snippet['resources']['firewall_rule']['properties']['protocol'] = 'any'
    stack = utils.parse_stack(snippet)
    rsrc = stack['firewall_rule']
    scheduler.TaskRunner(rsrc.create)()
    self.assertEqual((rsrc.CREATE, rsrc.COMPLETE), rsrc.state)
    self.mockclient.create_firewall_rule.assert_called_once_with({'firewall_rule': {'name': 'test-firewall-rule', 'shared': True, 'action': 'allow', 'protocol': None, 'enabled': True, 'ip_version': '4'}})