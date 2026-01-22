import copy
from unittest import mock
from oslo_config import cfg
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import nova
from heat.engine import node_data
from heat.engine.resources.aws.lb import loadbalancer as lb
from heat.engine import rsrc_defn
from heat.tests import common
from heat.tests.openstack.nova import fakes as fakes_nova
from heat.tests import utils
def test_child_params_default_sec_gr(self):
    template = template_format.parse(lb_template)
    del template['Parameters']['KeyName']
    del template['Parameters']['LbFlavor']
    del template['Resources']['LoadBalancer']['Properties']['SecurityGroups']
    del template['Parameters']['LbImageId']
    stack = utils.parse_stack(template)
    resource_name = 'LoadBalancer'
    lb_defn = stack.t.resource_definitions(stack)[resource_name]
    rsrc = lb.LoadBalancer(resource_name, lb_defn, stack)
    params = rsrc.child_params()
    expected = {'SecurityGroups': None}
    self.assertEqual(expected, params)