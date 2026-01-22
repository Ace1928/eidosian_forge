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
def test_plain_matches(self):
    resources = {u'a': {u'OS::Fruit': u'apples.yaml', u'restricted_actions': [u'update', u'replace']}, u'b': {u'OS::Food': u'fruity.yaml'}, u'nested': {u'res': {u'restricted_actions': 'update'}}}
    registry = environment.ResourceRegistry(None, {})
    registry.load({u'OS::Fruit': u'apples.yaml', 'resources': resources})
    self.assertIn(environment.UPDATE, registry.get_rsrc_restricted_actions('a'))
    self.assertNotIn(environment.UPDATE, registry.get_rsrc_restricted_actions('b'))
    self.assertNotIn(environment.UPDATE, registry.get_rsrc_restricted_actions('OS::Fruit'))
    self.assertNotIn(environment.UPDATE, registry.get_rsrc_restricted_actions('res'))
    self.assertNotIn(environment.UPDATE, registry.get_rsrc_restricted_actions('unknown'))