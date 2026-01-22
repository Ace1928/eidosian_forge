from unittest import mock
from neutronclient.common import exceptions as qe
from neutronclient.neutron import v2_0 as neutronV20
from neutronclient.v2_0 import client as neutronclient
from oslo_serialization import jsonutils
from heat.common import exception
from heat.common import template_format
from heat.engine import resource
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
def test_port_get_live_state(self):
    t = template_format.parse(neutron_port_template)
    t['resources']['port']['properties']['value_specs'] = {'binding:vif_type': 'test'}
    stack = utils.parse_stack(t)
    port = stack['port']
    resp = {'port': {'status': 'DOWN', 'binding:host_id': '', 'name': 'flip-port-xjbal77qope3', 'allowed_address_pairs': [], 'admin_state_up': True, 'network_id': 'd6859535-efef-4184-b236-e5fcae856e0f', 'dns_name': '', 'extra_dhcp_opts': [], 'mac_address': 'fa:16:3e:fe:64:79', 'qos_policy_id': 'some', 'dns_assignment': [], 'binding:vif_details': {}, 'binding:vif_type': 'unbound', 'device_owner': '', 'tenant_id': '30f466e3d14b4251853899f9c26e2b66', 'binding:profile': {}, 'port_security_enabled': True, 'propagate_uplink_status': True, 'binding:vnic_type': 'normal', 'fixed_ips': [{'subnet_id': '02d9608f-8f30-4611-ad02-69855c82457f', 'ip_address': '10.0.3.4'}], 'id': '829bf5c1-b59c-40ad-80e3-ea15a93879f3', 'security_groups': ['c276247f-50fd-4289-862a-80fb81a55de1'], 'device_id': ''}}
    port.client().show_port = mock.MagicMock(return_value=resp)
    port.resource_id = '1234'
    port._data = {}
    port.data_set = mock.Mock()
    reality = port.get_live_state(port.properties)
    expected = {'allowed_address_pairs': [], 'admin_state_up': True, 'device_owner': '', 'port_security_enabled': True, 'propagate_uplink_status': True, 'binding:vnic_type': 'normal', 'fixed_ips': [{'subnet': '02d9608f-8f30-4611-ad02-69855c82457f', 'ip_address': '10.0.3.4'}], 'security_groups': ['c276247f-50fd-4289-862a-80fb81a55de1'], 'device_id': '', 'dns_name': '', 'qos_policy': 'some', 'value_specs': {'binding:vif_type': 'unbound'}}
    self.assertEqual(set(expected.keys()), set(reality.keys()))
    for key in expected:
        self.assertEqual(expected[key], reality[key])