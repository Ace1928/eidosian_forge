from copy import deepcopy
from oslo_utils import uuidutils
from octavia_lib.api.drivers import data_models
from octavia_lib.common import constants
from octavia_lib.tests.unit import base
def test_to_dict_recursive_render_unset(self):
    ref_lb = data_models.LoadBalancer(admin_state_up=False, description='One great load balancer', flavor={'cake': 'chocolate'}, listeners=[self.ref_listener], loadbalancer_id=self.loadbalancer_id, project_id=self.project_id, vip_address=self.vip_address, vip_network_id=self.vip_network_id, vip_port_id=self.vip_port_id, vip_subnet_id=self.vip_subnet_id, vip_qos_policy_id=self.vip_qos_policy_id, availability_zone=self.availability_zone)
    ref_lb_dict_with_listener = deepcopy(self.ref_lb_dict_with_listener)
    ref_lb_dict_with_listener['pools'] = None
    ref_lb_dict_with_listener['name'] = None
    ref_lb_dict_with_listener['additional_vips'] = None
    ref_lb_converted_to_dict = ref_lb.to_dict(recurse=True, render_unsets=True)
    self.assertEqual(ref_lb_dict_with_listener, ref_lb_converted_to_dict)