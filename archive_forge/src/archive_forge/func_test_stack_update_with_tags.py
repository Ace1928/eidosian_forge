from unittest import mock
import uuid
import eventlet.queue
from oslo_config import cfg
from oslo_messaging import conffixture
from oslo_messaging.rpc import dispatcher
from heat.common import context
from heat.common import environment_util as env_util
from heat.common import exception
from heat.common import messaging
from heat.common import service_utils
from heat.common import template_format
from heat.db import api as db_api
from heat.engine.clients.os import glance
from heat.engine.clients.os import nova
from heat.engine.clients.os import swift
from heat.engine import environment
from heat.engine import resource
from heat.engine import service
from heat.engine import stack
from heat.engine import stack_lock
from heat.engine import template as templatem
from heat.objects import stack as stack_object
from heat.rpc import api as rpc_api
from heat.tests import common
from heat.tests.engine import tools
from heat.tests import utils
def test_stack_update_with_tags(self):
    """Test case for updating stack with tags.

        Create a stack with tags, then update with/without
        rpc_api.PARAM_EXISTING.
        """
    stack_name = 'service_update_test_stack_existing_tags'
    api_args = {rpc_api.PARAM_TIMEOUT: 60, rpc_api.PARAM_EXISTING: True}
    t = template_format.parse(tools.wp_template)
    stk = utils.parse_stack(t, stack_name=stack_name, tags=['tag1'])
    stk.set_stack_user_project_id('1234')
    self.assertEqual(['tag1'], stk.tags)
    self.patchobject(stack.Stack, 'validate')
    _, _, updated_stack = self.man._prepare_stack_updates(self.ctx, stk, t, {}, None, None, None, api_args, None)
    self.assertEqual(['tag1'], updated_stack.tags)
    api_args[rpc_api.STACK_TAGS] = []
    _, _, updated_stack = self.man._prepare_stack_updates(self.ctx, stk, t, {}, None, None, None, api_args, None)
    self.assertEqual([], updated_stack.tags)
    api_args[rpc_api.STACK_TAGS] = ['tag2']
    _, _, updated_stack = self.man._prepare_stack_updates(self.ctx, stk, t, {}, None, None, None, api_args, None)
    self.assertEqual(['tag2'], updated_stack.tags)
    api_args[rpc_api.STACK_TAGS] = ['tag3']
    _, _, updated_stack = self.man._prepare_stack_updates(self.ctx, stk, t, {}, None, None, None, api_args, None)
    self.assertEqual(['tag3'], updated_stack.tags)
    del api_args[rpc_api.PARAM_EXISTING]
    del api_args[rpc_api.STACK_TAGS]
    _, _, updated_stack = self.man._prepare_stack_updates(self.ctx, stk, t, {}, None, None, None, api_args, None)
    self.assertEqual([], updated_stack.tags)