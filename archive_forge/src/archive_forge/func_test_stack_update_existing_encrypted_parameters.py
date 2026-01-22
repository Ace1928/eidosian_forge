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
def test_stack_update_existing_encrypted_parameters(self):
    hidden_param_template = u'\nheat_template_version: 2013-05-23\nparameters:\n   param2:\n     type: string\n     description: value2.\n     hidden: true\nresources:\n   a_resource:\n       type: GenericResourceType\n'
    cfg.CONF.set_override('encrypt_parameters_and_properties', True)
    stack_name = 'service_update_test_stack_encrypted_parameters'
    t = template_format.parse(hidden_param_template)
    env1 = environment.Environment({'param2': 'bar'})
    stk = stack.Stack(self.ctx, stack_name, templatem.Template(t, env=env1))
    stk.store()
    stk.set_stack_user_project_id('1234')
    db_tpl = db_api.raw_template_get(self.ctx, stk.t.id)
    db_params = db_tpl.environment['parameters']
    self.assertEqual('cryptography_decrypt_v1', db_params['param2'][0])
    self.assertNotEqual('foo', db_params['param2'][1])
    loaded_stack = stack.Stack.load(self.ctx, stack_id=stk.id)
    params = loaded_stack.t.env.params
    self.assertEqual('bar', params.get('param2'))
    update_params = {'encrypted_param_names': [], 'parameter_defaults': {}, 'event_sinks': [], 'parameters': {}, 'resource_registry': {'resources': {}}}
    api_args = {rpc_api.PARAM_TIMEOUT: 60, rpc_api.PARAM_EXISTING: True, rpc_api.PARAM_CONVERGE: False}
    with mock.patch('heat.engine.stack.Stack') as mock_stack:
        loaded_stack.update = mock.Mock()
        self.patchobject(service, 'NotifyEvent')
        mock_stack.load.return_value = loaded_stack
        mock_stack.validate.return_value = None
        result = self.man.update_stack(self.ctx, stk.identifier(), t, update_params, None, api_args)
        tmpl = mock_stack.call_args[0][2]
        self.assertEqual({u'param2': u'bar'}, tmpl.env.params)
        self.assertEqual([u'param2'], tmpl.env.encrypted_param_names)
        self.assertEqual(stk.identifier(), result)