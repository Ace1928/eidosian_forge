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
def test_global_registry(self):
    self.g_env.register_class('CloudX::Nova::Server', generic_resource.GenericResource)
    new_env = {u'parameters': {u'a': u'ff', u'b': u'ss'}, u'resource_registry': {u'OS::*': 'CloudX::*'}}
    env = environment.Environment(new_env)
    self.assertEqual('CloudX::Nova::Server', env.get_resource_info('OS::Nova::Server', 'my_db_server').name)