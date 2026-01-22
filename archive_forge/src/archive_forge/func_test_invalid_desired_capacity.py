import copy
from unittest import mock
from oslo_config import cfg
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import senlin
from heat.engine.resources.openstack.senlin import cluster as sc
from heat.engine import scheduler
from heat.engine import template
from heat.tests import common
from heat.tests import utils
from openstack import exceptions
def test_invalid_desired_capacity(self):
    self.t['resources']['senlin-cluster']['properties']['min_size'] = 1
    self.t['resources']['senlin-cluster']['properties']['max_size'] = 2
    self.t['resources']['senlin-cluster']['properties']['desired_capacity'] = 3
    stack = utils.parse_stack(self.t)
    ex = self.assertRaises(exception.StackValidationFailed, stack['senlin-cluster'].validate)
    self.assertEqual('desired_capacity must be between min_size and max_size', str(ex))