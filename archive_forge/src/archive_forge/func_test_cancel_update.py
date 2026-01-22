import json
import os
from unittest import mock
from oslo_config import fixture as config_fixture
from heat.api.aws import exception
import heat.api.cfn.v1.stacks as stacks
from heat.common import exception as heat_exception
from heat.common import identifier
from heat.common import policy
from heat.common import wsgi
from heat.rpc import api as rpc_api
from heat.rpc import client as rpc_client
from heat.tests import common
from heat.tests import utils
def test_cancel_update(self):
    stack_name = 'wordpress'
    params = {'Action': 'CancelUpdateStack', 'StackName': stack_name}
    dummy_req = self._dummy_GET_request(params)
    self._stub_enforce(dummy_req, 'CancelUpdateStack')
    identity = dict(identifier.HeatIdentifier('t', stack_name, '1'))
    self.m_call.return_value = identity
    response = self.controller.cancel_update(dummy_req)
    expected = {'CancelUpdateStackResponse': {'CancelUpdateStackResult': {}}}
    self.assertEqual(response, expected)
    self.assertEqual([mock.call(dummy_req.context, ('identify_stack', {'stack_name': stack_name})), mock.call(dummy_req.context, ('stack_cancel_update', {'stack_identity': identity, 'cancel_with_rollback': True}), version='1.14')], self.m_call.call_args_list)