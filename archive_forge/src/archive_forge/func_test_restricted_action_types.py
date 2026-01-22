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
def test_restricted_action_types(self):
    resources = {u'update': {u'restricted_actions': 'update'}, u'replace': {u'restricted_actions': 'replace'}, u'all': {u'restricted_actions': ['update', 'replace']}}
    registry = environment.ResourceRegistry(None, {})
    registry.load({'resources': resources})
    self.assertIn(environment.UPDATE, registry.get_rsrc_restricted_actions('update'))
    self.assertNotIn(environment.UPDATE, registry.get_rsrc_restricted_actions('replace'))
    self.assertIn(environment.REPLACE, registry.get_rsrc_restricted_actions('replace'))
    self.assertNotIn(environment.REPLACE, registry.get_rsrc_restricted_actions('update'))
    self.assertIn(environment.UPDATE, registry.get_rsrc_restricted_actions('all'))
    self.assertIn(environment.REPLACE, registry.get_rsrc_restricted_actions('all'))