import os.path
from unittest import mock
import fixtures
from oslo_config import cfg
from heat.common import environment_format
from heat.common import exception
from heat.engine import environment
from heat.engine import resources
from heat.engine.resources.aws.ec2 import instance
from heat.engine.resources.openstack.nova import server
from heat.engine import support
from heat.tests import common
from heat.tests import generic_resource
from heat.tests import utils
def test_params_parent_overwritten(self):
    new_params = {'parameters': {'foo': 'bar', 'tester': 'Yes'}}
    parent_params = {'parameters': {'gone': 'hopefully'}}
    penv = environment.Environment(env=parent_params)
    expected = {'parameter_defaults': {}, 'encrypted_param_names': [], 'event_sinks': [], 'resource_registry': {'resources': {}}}
    expected.update(new_params)
    cenv = environment.get_child_environment(penv, new_params)
    self.assertEqual(expected, cenv.env_as_dict())