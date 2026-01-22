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
def test_resources_load(self):
    resources = {u'pre_create': {u'OS::Fruit': u'apples.yaml', u'hooks': 'pre-create'}, u'pre_update': {u'hooks': 'pre-update'}, u'both': {u'hooks': ['pre-create', 'pre-update']}, u'b': {u'OS::Food': u'fruity.yaml'}, u'nested': {u'res': {u'hooks': 'pre-create'}}}
    registry = environment.ResourceRegistry(None, {})
    registry.load({'resources': resources})
    self.assertIsNotNone(registry.get_resource_info('OS::Fruit', resource_name='pre_create'))
    self.assertIsNotNone(registry.get_resource_info('OS::Food', resource_name='b'))
    resources = registry.as_dict()['resources']
    self.assertEqual('pre-create', resources['pre_create']['hooks'])
    self.assertEqual('pre-update', resources['pre_update']['hooks'])
    self.assertEqual(['pre-create', 'pre-update'], resources['both']['hooks'])
    self.assertEqual('pre-create', resources['nested']['res']['hooks'])