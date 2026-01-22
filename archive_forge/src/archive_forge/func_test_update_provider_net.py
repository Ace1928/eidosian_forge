import copy
from unittest import mock
from neutronclient.common import exceptions as qe
from neutronclient.v2_0 import client as neutronclient
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import neutron
from heat.engine.resources.openstack.neutron import provider_net
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
def test_update_provider_net(self):
    resource_type = 'networks'
    rsrc = self.create_provider_net()
    self.mockclient.show_network.side_effect = [stpnb, stpna]
    self.mockclient.update_network.return_value = None
    rsrc.validate()
    scheduler.TaskRunner(rsrc.create)()
    self.assertEqual((rsrc.CREATE, rsrc.COMPLETE), rsrc.state)
    prop_diff = {'network_type': 'vlan', 'physical_network': 'physnet_1', 'segmentation_id': '102', 'port_security_enabled': False, 'router_external': True, 'tags': []}
    update_snippet = rsrc_defn.ResourceDefinition(rsrc.name, rsrc.type(), prop_diff)
    self.assertIsNone(rsrc.handle_update(update_snippet, {}, prop_diff))
    self.assertIsNone(rsrc.handle_update(update_snippet, {}, {'name': None}))
    self.assertIsNone(rsrc.handle_update(update_snippet, {}, {}))
    self.mockclient.create_network.assert_called_once_with({'network': {'name': u'the_provider_network', 'admin_state_up': True, 'provider:network_type': 'vlan', 'provider:physical_network': 'physnet_1', 'provider:segmentation_id': '101', 'router:external': False, 'shared': True, 'availability_zone_hints': ['az1']}})
    self.mockclient.replace_tag.assert_called_with(resource_type, 'fc68ea2c-b60b-4b4f-bd82-94ec81110766', {'tags': []})
    self.mockclient.show_network.assert_called_with('fc68ea2c-b60b-4b4f-bd82-94ec81110766')
    self.assertEqual(2, self.mockclient.show_network.call_count)
    self.mockclient.update_network.assert_has_calls([mock.call('fc68ea2c-b60b-4b4f-bd82-94ec81110766', {'network': {'provider:network_type': 'vlan', 'provider:physical_network': 'physnet_1', 'provider:segmentation_id': '102', 'port_security_enabled': False, 'router:external': True}}), mock.call('fc68ea2c-b60b-4b4f-bd82-94ec81110766', {'network': {'name': utils.PhysName(rsrc.stack.name, 'provider_net')}})])
    self.assertEqual(2, self.mockclient.update_network.call_count)