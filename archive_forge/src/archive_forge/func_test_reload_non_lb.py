from unittest import mock
from heat.common import exception
from heat.common import template_format
from heat.engine import properties
from heat.engine import resource
from heat.scaling import lbutils
from heat.tests import common
from heat.tests import generic_resource
from heat.tests import utils
def test_reload_non_lb(self):
    id_list = ['ID1', 'ID2', 'ID3']
    non_lb = self.stack['non_lb']
    error = self.assertRaises(exception.Error, lbutils.reconfigure_loadbalancers, [non_lb], id_list)
    self.assertIn("Unsupported resource 'non_lb' in LoadBalancerNames", str(error))