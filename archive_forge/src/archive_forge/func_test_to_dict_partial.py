from copy import deepcopy
from oslo_utils import uuidutils
from octavia_lib.api.drivers import data_models
from octavia_lib.common import constants
from octavia_lib.tests.unit import base
def test_to_dict_partial(self):
    ref_lb = data_models.LoadBalancer(loadbalancer_id=self.loadbalancer_id)
    ref_lb_dict = {'loadbalancer_id': self.loadbalancer_id}
    ref_lb_converted_to_dict = ref_lb.to_dict()
    self.assertEqual(ref_lb_dict, ref_lb_converted_to_dict)