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
def test_loadbalancer_validate_hchk_good(self):
    hc = {'Target': 'HTTP:80/', 'HealthyThreshold': '3', 'UnhealthyThreshold': '5', 'Interval': '30', 'Timeout': '5'}
    rsrc = self.setup_loadbalancer(hc=hc)
    rsrc._parse_nested_stack = mock.Mock()
    self.assertIsNone(rsrc.validate())