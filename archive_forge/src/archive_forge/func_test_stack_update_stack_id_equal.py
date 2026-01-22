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
def test_stack_update_stack_id_equal(self):
    stack_name = 'test_stack_update_stack_id_equal'
    tpl = {'HeatTemplateFormatVersion': '2012-12-12', 'Resources': {'A': {'Type': 'ResourceWithPropsType', 'Properties': {'Foo': {'Ref': 'AWS::StackId'}}}}}
    template = templatem.Template(tpl)
    create_stack = stack.Stack(self.ctx, stack_name, template)
    sid = create_stack.store()
    create_stack.create()
    self.assertEqual((create_stack.CREATE, create_stack.COMPLETE), create_stack.state)
    create_stack._persist_state()
    s = stack_object.Stack.get_by_id(self.ctx, sid)
    old_stack = stack.Stack.load(self.ctx, stack=s)
    self.assertEqual((old_stack.CREATE, old_stack.COMPLETE), old_stack.state)
    self.assertEqual(create_stack.identifier().arn(), old_stack['A'].properties['Foo'])
    mock_load = self.patchobject(stack.Stack, 'load', return_value=old_stack)
    result = self.man.update_stack(self.ctx, create_stack.identifier(), tpl, {}, None, {rpc_api.PARAM_CONVERGE: False})
    old_stack._persist_state()
    self.assertEqual((old_stack.UPDATE, old_stack.COMPLETE), old_stack.state)
    self.assertEqual(create_stack.identifier(), result)
    self.assertIsNotNone(create_stack.identifier().stack_id)
    self.assertEqual(create_stack.identifier().arn(), old_stack['A'].properties['Foo'])
    self.assertEqual(create_stack['A'].id, old_stack['A'].id)
    mock_load.assert_called_once_with(self.ctx, stack=s, check_refresh_cred=True)