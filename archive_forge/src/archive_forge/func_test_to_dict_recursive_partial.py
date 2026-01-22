from copy import deepcopy
from oslo_utils import uuidutils
from octavia_lib.api.drivers import data_models
from octavia_lib.common import constants
from octavia_lib.tests.unit import base
def test_to_dict_recursive_partial(self):
    ref_lb = data_models.LoadBalancer(loadbalancer_id=self.loadbalancer_id, listeners=[self.ref_listener])
    ref_lb_dict_with_listener = {'loadbalancer_id': self.loadbalancer_id, 'listeners': [self.ref_listener_dict]}
    ref_lb_dict_with_listener = deepcopy(ref_lb_dict_with_listener)
    ref_lb_dict_with_listener['listeners'][0].pop('description', None)
    ref_lb_converted_to_dict = ref_lb.to_dict(recurse=True)
    self.assertEqual(ref_lb_dict_with_listener, ref_lb_converted_to_dict)