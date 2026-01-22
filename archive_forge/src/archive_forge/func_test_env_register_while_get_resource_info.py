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
def test_env_register_while_get_resource_info(self):
    env_test = {u'resource_registry': {u'OS::Test::Dummy': self.resource_type}}
    env = environment.Environment()
    env.load(env_test)
    env.get_resource_info('OS::Test::Dummy')
    self.assertEqual({'OS::Test::Dummy': self.resource_type, 'resources': {}}, env.user_env_as_dict().get(environment_format.RESOURCE_REGISTRY))
    env_test = {u'resource_registry': {u'resources': {u'test': {u'OS::Test::Dummy': self.resource_type}}}}
    env.load(env_test)
    env.get_resource_info('OS::Test::Dummy')
    self.assertEqual({u'OS::Test::Dummy': self.resource_type, 'resources': {u'test': {u'OS::Test::Dummy': self.resource_type}}}, env.user_env_as_dict().get(environment_format.RESOURCE_REGISTRY))