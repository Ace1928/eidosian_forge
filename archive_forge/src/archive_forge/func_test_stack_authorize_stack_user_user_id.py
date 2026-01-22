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
def test_stack_authorize_stack_user_user_id(self):
    self.ctx = utils.dummy_context(user_id=str(uuid.uuid4()))
    stack_name = 'stack_authorize_stack_user_user_id'
    stack = tools.get_stack(stack_name, self.ctx, server_config_template)
    self.stack = stack

    def handler(resource_name):
        return resource_name == 'WebServer'
    self.stack.register_access_allowed_handler(self.ctx.user_id, handler)
    self.assertTrue(self.eng._authorize_stack_user(self.ctx, self.stack, 'WebServer'))
    self.assertFalse(self.eng._authorize_stack_user(self.ctx, self.stack, 'NoSuchResource'))
    self.ctx.user = str(uuid.uuid4())
    self.assertFalse(self.eng._authorize_stack_user(self.ctx, self.stack, 'WebServer'))