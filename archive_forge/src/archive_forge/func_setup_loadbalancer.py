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
def setup_loadbalancer(self, include_magic=True, cache_data=None, hc=None):
    template = template_format.parse(lb_template)
    if not include_magic:
        del template['Parameters']['KeyName']
        del template['Parameters']['LbFlavor']
        del template['Parameters']['LbImageId']
    if hc is not None:
        props = template['Resources']['LoadBalancer']['Properties']
        props['HealthCheck'] = hc
    self.stack = utils.parse_stack(template, cache_data=cache_data)
    resource_name = 'LoadBalancer'
    lb_defn = self.stack.defn.resource_definition(resource_name)
    return lb.LoadBalancer(resource_name, lb_defn, self.stack)