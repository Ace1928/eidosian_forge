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
def test_drill_down_to_child_resource(self):
    env = {u'resource_registry': {u'OS::Food': u'fruity.yaml', u'resources': {u'a': {u'OS::Fruit': u'apples.yaml', u'hooks': 'pre-create'}, u'nested': {u'b': {u'OS::Fruit': u'carrots.yaml'}, u'nested_res': {u'hooks': 'pre-create'}}}}}
    penv = environment.Environment(env)
    cenv = environment.get_child_environment(penv, None, child_resource_name=u'nested')
    registry = cenv.user_env_as_dict()['resource_registry']
    resources = registry['resources']
    self.assertIn('nested_res', resources)
    self.assertIn('hooks', resources['nested_res'])
    self.assertIsNotNone(cenv.get_resource_info('OS::Food', resource_name='abc'))
    self.assertRaises(exception.EntityNotFound, cenv.get_resource_info, 'OS::Fruit', resource_name='a')
    res = cenv.get_resource_info('OS::Fruit', resource_name='b')
    self.assertIsNotNone(res)
    self.assertEqual(u'carrots.yaml', res.value)