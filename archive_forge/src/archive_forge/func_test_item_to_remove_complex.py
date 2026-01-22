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
def test_item_to_remove_complex(self):
    env = {u'resource_registry': {u'OS::Food': u'fruity.yaml', u'resources': {u'abc': {u'OS::Food': u'nutty.yaml'}}}}
    penv = environment.Environment(env)
    victim = penv.get_resource_info('OS::Food', resource_name='abc')
    self.assertEqual(['resources', 'abc', 'OS::Food'], victim.path)
    cenv = environment.get_child_environment(penv, None, item_to_remove=victim)
    res = cenv.get_resource_info('OS::Food', resource_name='abc')
    self.assertEqual(['OS::Food'], res.path)
    rr = cenv.user_env_as_dict()['resource_registry']
    self.assertIn('OS::Food', rr)
    self.assertNotIn('OS::Food', rr['resources']['abc'])
    innocent2 = penv.get_resource_info('OS::Food', resource_name='abc')
    self.assertEqual(['resources', 'abc', 'OS::Food'], innocent2.path)