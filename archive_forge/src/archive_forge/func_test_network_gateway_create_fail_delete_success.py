from unittest import mock
from neutronclient.common import exceptions as qe
from neutronclient.neutron import v2_0 as neutronV20
from neutronclient.v2_0 import client as neutronclient
from heat.common import exception
from heat.common import template_format
from heat.common import timeutils
from heat.engine.resources.openstack.neutron import network_gateway
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
def test_network_gateway_create_fail_delete_success(self):
    rsrc = self.mock_create_fail_network_not_found_delete_success()
    self.stub_NetworkConstraint_validate()
    rsrc.validate()
    self.assertRaises(exception.ResourceFailure, scheduler.TaskRunner(rsrc.create))
    self.assertEqual((rsrc.CREATE, rsrc.FAILED), rsrc.state)
    ref_id = rsrc.FnGetRefId()
    self.assertEqual(u'ed4c03b9-8251-4c09-acc4-e59ee9e6aa37', ref_id)
    self.assertIsNone(scheduler.TaskRunner(rsrc.delete)())
    self.assertEqual((rsrc.DELETE, rsrc.COMPLETE), rsrc.state)
    self.mockclient.create_network_gateway.assert_called_once_with({'network_gateway': {'name': 'NetworkGateway', 'devices': [{'id': 'e52148ca-7db9-4ec3-abe6-2c7c0ff316eb', 'interface_name': 'breth1'}]}})
    self.mockclient.connect_network_gateway.assert_called_once_with('ed4c03b9-8251-4c09-acc4-e59ee9e6aa37', {'network_id': '6af055d3-26f6-48dd-a597-7611d7e58d35', 'segmentation_id': 10, 'segmentation_type': 'vlan'})
    self.mockclient.disconnect_network_gateway.assert_called_once_with('ed4c03b9-8251-4c09-acc4-e59ee9e6aa37', {'network_id': '6af055d3-26f6-48dd-a597-7611d7e58d35', 'segmentation_id': 10, 'segmentation_type': 'vlan'})
    self.mockclient.delete_network_gateway.assert_called_once_with('ed4c03b9-8251-4c09-acc4-e59ee9e6aa37')
    self.mockclient.show_network_gateway.assert_called_once_with('ed4c03b9-8251-4c09-acc4-e59ee9e6aa37')