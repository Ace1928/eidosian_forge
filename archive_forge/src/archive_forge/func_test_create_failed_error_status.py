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
def test_create_failed_error_status(self):
    cfg.CONF.set_override('action_retry_limit', 0)
    rsrc = self.create_firewall()
    self.mockclient.show_firewall.side_effect = [{'firewall': {'status': 'PENDING_CREATE'}}, {'firewall': {'status': 'ERROR'}}]
    error = self.assertRaises(exception.ResourceFailure, scheduler.TaskRunner(rsrc.create))
    self.assertEqual('ResourceInError: resources.firewall: Went to status ERROR due to "Error in Firewall"', str(error))
    self.assertEqual((rsrc.CREATE, rsrc.FAILED), rsrc.state)
    self.mockclient.create_firewall.assert_called_once_with({'firewall': {'name': 'test-firewall', 'admin_state_up': True, 'router_ids': ['router_1', 'router_2'], 'firewall_policy_id': 'policy-id', 'shared': True}})
    self.mockclient.show_firewall.assert_called_with('5678')