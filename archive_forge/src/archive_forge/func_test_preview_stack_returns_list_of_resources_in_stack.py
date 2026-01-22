from unittest import mock
import uuid
from oslo_config import cfg
from oslo_messaging.rpc import dispatcher
from oslo_serialization import jsonutils as json
from heat.common import context
from heat.common import environment_util as env_util
from heat.common import exception
from heat.common import identifier
from heat.common import policy
from heat.common import template_format
from heat.engine.cfn import template as cfntemplate
from heat.engine import environment
from heat.engine.hot import functions as hot_functions
from heat.engine.hot import template as hottemplate
from heat.engine import resource as res
from heat.engine import service
from heat.engine import stack as parser
from heat.engine import template as templatem
from heat.objects import stack as stack_object
from heat.rpc import api as rpc_api
from heat.tests import common
from heat.tests.engine import tools
from heat.tests import generic_resource as generic_rsrc
from heat.tests.openstack.nova import fakes as fakes_nova
from heat.tests import utils
def test_preview_stack_returns_list_of_resources_in_stack(self):
    stack = self._preview_stack()
    self.assertIsInstance(stack['resources'], list)
    self.assertEqual(2, len(stack['resources']))
    resource_types = set((r['resource_type'] for r in stack['resources']))
    self.assertIn('GenericResource1', resource_types)
    self.assertIn('GenericResource2', resource_types)
    resource_names = set((r['resource_name'] for r in stack['resources']))
    self.assertIn('SampleResource1', resource_names)
    self.assertIn('SampleResource2', resource_names)