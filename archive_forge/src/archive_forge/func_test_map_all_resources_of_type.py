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
def test_map_all_resources_of_type(self):
    self.g_env.register_class('OS::Nova::FloatingIP', generic_resource.GenericResource)
    new_env = {u'parameters': {u'a': u'ff', u'b': u'ss'}, u'resource_registry': {u'OS::Networking::FloatingIP': 'OS::Nova::FloatingIP', u'OS::Loadbalancer': 'lb.yaml'}}
    env = environment.Environment(new_env)
    self.assertEqual('OS::Nova::FloatingIP', env.get_resource_info('OS::Networking::FloatingIP', 'my_fip').name)