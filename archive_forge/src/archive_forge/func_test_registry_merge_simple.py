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
def test_registry_merge_simple(self):
    env1 = {u'resource_registry': {u'OS::Food': u'fruity.yaml'}}
    env2 = {u'resource_registry': {u'OS::Fruit': u'apples.yaml'}}
    penv = environment.Environment(env=env1)
    cenv = environment.get_child_environment(penv, env2)
    rr = cenv.user_env_as_dict()['resource_registry']
    self.assertIn('OS::Food', rr)
    self.assertIn('OS::Fruit', rr)