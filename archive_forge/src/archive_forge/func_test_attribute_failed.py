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
def test_attribute_failed(self):
    rsrc = self.create_firewall_rule()
    scheduler.TaskRunner(rsrc.create)()
    error = self.assertRaises(exception.InvalidTemplateAttribute, rsrc.FnGetAtt, 'subnet_id')
    self.assertEqual('The Referenced Attribute (firewall_rule subnet_id) is incorrect.', str(error))
    self.mockclient.create_firewall_rule.assert_called_once_with({'firewall_rule': {'name': 'test-firewall-rule', 'shared': True, 'action': 'allow', 'protocol': 'tcp', 'enabled': True, 'ip_version': '4'}})
    self.mockclient.show_firewall_rule.assert_not_called()