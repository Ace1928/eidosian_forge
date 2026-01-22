import copy
from unittest import mock
import uuid
from neutronclient.v2_0 import client as neutronclient
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import cinder
from heat.engine.clients.os import glance
from heat.engine.clients.os import neutron
from heat.engine.clients.os import nova
from heat.engine.clients import progress
from heat.engine import environment
from heat.engine import resource
from heat.engine.resources.aws.ec2 import instance as instances
from heat.engine.resources import scheduler_hints as sh
from heat.engine import scheduler
from heat.engine import stack as parser
from heat.engine import template
from heat.tests import common
from heat.tests.openstack.nova import fakes as fakes_nova
from heat.tests import utils
def test_instance_update_network_interfaces_no_old(self):
    """Test case for updating NetworkInterfaces when there's no old prop.

        Instance.handle_update supports changing the NetworkInterfaces,
        and makes the change making a resize API call against Nova.
        """
    return_server = self.fc.servers.list()[1]
    return_server.id = '1234'
    instance = self._create_test_instance(return_server, 'ud_network_interfaces')
    self._stub_glance_for_update()
    new_interfaces = [{'NetworkInterfaceId': 'ea29f957-cd35-4364-98fb-57ce9732c10d', 'DeviceIndex': '2'}, {'NetworkInterfaceId': '34b752ec-14de-416a-8722-9531015e04a5', 'DeviceIndex': '3'}]
    iface = self.create_fake_iface('d1e9c73c-04fe-4e9e-983c-d5ef94cd1a46', 'c4485ba1-283a-4f5f-8868-0cd46cdda52f', '10.0.0.4')
    update_props = self.instance_props.copy()
    update_props['NetworkInterfaces'] = new_interfaces
    update_template = instance.t.freeze(properties=update_props)
    self.fc.servers.get = mock.Mock(return_value=return_server)
    return_server.interface_list = mock.Mock(return_value=[iface])
    return_server.interface_detach = mock.Mock(return_value=None)
    return_server.interface_attach = mock.Mock(return_value=None)
    scheduler.TaskRunner(instance.update, update_template)()
    self.assertEqual((instance.UPDATE, instance.COMPLETE), instance.state)
    self.fc.servers.get.assert_called_with('1234')
    return_server.interface_list.assert_called_once_with()
    return_server.interface_detach.assert_called_once_with('d1e9c73c-04fe-4e9e-983c-d5ef94cd1a46')
    return_server.interface_attach.assert_has_calls([mock.call('ea29f957-cd35-4364-98fb-57ce9732c10d', None, None), mock.call('34b752ec-14de-416a-8722-9531015e04a5', None, None)])
    self.assertEqual(2, return_server.interface_attach.call_count)