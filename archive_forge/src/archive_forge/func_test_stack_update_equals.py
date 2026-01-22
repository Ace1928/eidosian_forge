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
@mock.patch.object(stack_object.Stack, 'count_total_resources')
def test_stack_update_equals(self, ctr):
    stack_name = 'test_stack_update_equals_resource_limit'
    params = {}
    tpl = {'HeatTemplateFormatVersion': '2012-12-12', 'Resources': {'A': {'Type': 'GenericResourceType'}, 'B': {'Type': 'GenericResourceType'}, 'C': {'Type': 'GenericResourceType'}}}
    template = templatem.Template(tpl)
    old_stack = stack.Stack(self.ctx, stack_name, template)
    sid = old_stack.store()
    old_stack.set_stack_user_project_id('1234')
    s = stack_object.Stack.get_by_id(self.ctx, sid)
    ctr.return_value = 3
    stk = stack.Stack(self.ctx, stack_name, template)
    mock_stack = self.patchobject(stack, 'Stack', return_value=stk)
    mock_load = self.patchobject(stack.Stack, 'load', return_value=old_stack)
    mock_tmpl = self.patchobject(templatem, 'Template', return_value=stk.t)
    mock_env = self.patchobject(environment, 'Environment', return_value=stk.env)
    mock_validate = self.patchobject(stk, 'validate', return_value=None)
    cfg.CONF.set_override('max_resources_per_stack', 3)
    api_args = {'timeout_mins': 60, rpc_api.PARAM_CONVERGE: False}
    result = self.man.update_stack(self.ctx, old_stack.identifier(), template, params, None, api_args)
    self.assertEqual(old_stack.identifier(), result)
    self.assertIsInstance(result, dict)
    self.assertTrue(result['stack_id'])
    root_stack_id = old_stack.root_stack_id()
    self.assertEqual(3, old_stack.total_resources(root_stack_id))
    mock_tmpl.assert_called_once_with(template, files=None)
    mock_env.assert_called_once_with(params)
    mock_stack.assert_called_once_with(self.ctx, stk.name, stk.t, convergence=False, current_traversal=old_stack.current_traversal, prev_raw_template_id=None, current_deps=None, disable_rollback=True, nested_depth=0, owner_id=None, parent_resource=None, stack_user_project_id='1234', strict_validate=True, tenant_id='test_tenant_id', timeout_mins=60, user_creds_id=u'1', username='test_username', converge=False)
    mock_load.assert_called_once_with(self.ctx, stack=s, check_refresh_cred=True)
    mock_validate.assert_called_once_with()