import copy
from unittest import mock
from neutronclient.common import exceptions as qe
from neutronclient.neutron import v2_0 as neutronV20
from neutronclient.v2_0 import client as neutronclient
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import neutron
from heat.engine.clients.os import openstacksdk
from heat.engine.hot import functions as hot_funcs
from heat.engine import node_data
from heat.engine import resource
from heat.engine.resources.openstack.neutron import subnet
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.engine import stk_defn
from heat.tests import common
from heat.tests import utils
def test_subnet(self):
    update_props = {'subnet': {'dns_nameservers': ['8.8.8.8', '192.168.1.254'], 'name': 'mysubnet', 'enable_dhcp': True, 'host_routes': [{'destination': '192.168.1.0/24', 'nexthop': '194.168.1.2'}], 'gateway_ip': '10.0.3.105', 'tags': ['tag2', 'tag3'], 'allocation_pools': [{'start': '10.0.3.20', 'end': '10.0.3.100'}, {'start': '10.0.3.110', 'end': '10.0.3.200'}]}}
    t, stack = self._setup_mock(tags=['tag1', 'tag2'])
    create_props = {'subnet': {'name': utils.PhysName(stack.name, 'test_subnet'), 'network_id': 'fc68ea2c-b60b-4b4f-bd82-94ec81110766', 'dns_nameservers': [u'8.8.8.8'], 'allocation_pools': [{'start': u'10.0.3.20', 'end': u'10.0.3.150'}], 'host_routes': [{'destination': u'10.0.4.0/24', 'nexthop': u'10.0.3.20'}], 'ip_version': 4, 'cidr': u'10.0.3.0/24', 'tenant_id': 'c1210485b2424d48804aad5d39c61b8f', 'enable_dhcp': True}}
    self.patchobject(stack['net'], 'FnGetRefId', return_value='fc68ea2c-b60b-4b4f-bd82-94ec81110766')
    set_tag_mock = self.patchobject(neutronclient.Client, 'replace_tag')
    rsrc = self.create_subnet(t, stack, 'sub_net')
    scheduler.TaskRunner(rsrc.create)()
    self.assertEqual((rsrc.CREATE, rsrc.COMPLETE), rsrc.state)
    self.create_mock.assert_called_once_with(create_props)
    set_tag_mock.assert_called_once_with('subnets', rsrc.resource_id, {'tags': ['tag1', 'tag2']})
    rsrc.validate()
    ref_id = rsrc.FnGetRefId()
    self.assertEqual('91e47a57-7508-46fe-afc9-fc454e8580e1', ref_id)
    self.assertIsNone(rsrc.FnGetAtt('network_id'))
    self.assertEqual('fc68ea2c-b60b-4b4f-bd82-94ec81110766', rsrc.FnGetAtt('network_id'))
    self.assertEqual('8.8.8.8', rsrc.FnGetAtt('dns_nameservers')[0])
    self.assertIn(stack['port'], stack.dependencies[stack['sub_net']])
    self.assertIn(stack['port2'], stack.dependencies[stack['sub_net']])
    update_snippet = rsrc_defn.ResourceDefinition(rsrc.name, rsrc.type(), update_props['subnet'])
    rsrc.handle_update(update_snippet, {}, update_props['subnet'])
    self.update_mock.assert_called_once_with('91e47a57-7508-46fe-afc9-fc454e8580e1', update_props)
    set_tag_mock.assert_called_with('subnets', rsrc.resource_id, {'tags': ['tag2', 'tag3']})
    del update_props['subnet']['name']
    rsrc.handle_update(update_snippet, {}, update_props['subnet'])
    self.update_mock.assert_called_with('91e47a57-7508-46fe-afc9-fc454e8580e1', update_props)
    rsrc.handle_update(update_snippet, {}, {})
    self.assertIsNone(scheduler.TaskRunner(rsrc.delete)())
    rsrc.state_set(rsrc.CREATE, rsrc.COMPLETE, 'to delete again')
    self.assertIsNone(scheduler.TaskRunner(rsrc.delete)())