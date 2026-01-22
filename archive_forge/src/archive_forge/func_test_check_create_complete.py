from unittest import mock
from neutronclient.neutron import v2_0 as neutronV20
from osc_lib import exceptions
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os.octavia import OctaviaClientPlugin
from heat.engine.resources.openstack.octavia import loadbalancer
from heat.tests import common
from heat.tests.openstack.octavia import inline_templates
from heat.tests import utils
def test_check_create_complete(self):
    self._create_stack()
    self.octavia_client.load_balancer_show.side_effect = [{'provisioning_status': 'ACTIVE'}, {'provisioning_status': 'PENDING_CREATE'}, {'provisioning_status': 'ERROR'}]
    self.assertTrue(self.lb.check_create_complete(None))
    self.assertFalse(self.lb.check_create_complete(None))
    self.assertRaises(exception.ResourceInError, self.lb.check_create_complete, None)