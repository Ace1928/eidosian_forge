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
def test_wordpress_single_instance_stack_adopt_fail(self):
    t = template_format.parse(tools.wp_template)
    template = templatem.Template(t)
    ctx = utils.dummy_context()
    adopt_data = {'resources': {'WebServer1': {'resource_id': 'test-res-id'}}}
    stack = parser.Stack(ctx, 'test_stack', template, adopt_stack_data=adopt_data)
    fc = tools.setup_mocks_with_mock(self, stack, mock_image_constraint=False)
    stack.store()
    stack.adopt()
    self.assertIsNotNone(stack['WebServer'])
    expected = 'Resource ADOPT failed: Exception: resources.WebServer: Resource ID was not provided.'
    self.assertEqual(expected, stack.status_reason)
    self.assertEqual((stack.ADOPT, stack.FAILED), stack.state)
    tools.validate_setup_mocks_with_mock(stack, fc, mock_image_constraint=False, validate_create=False)