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
def test_get_template_bad_name(self):
    stack_name = 'wibble'
    params = {'Action': 'GetTemplate', 'StackName': stack_name}
    dummy_req = self._dummy_GET_request(params)
    self._stub_enforce(dummy_req, 'GetTemplate')
    exc = heat_exception.EntityNotFound(entity='Stack', name='test')
    self.m_call.side_effect = [exc]
    result = self.controller.get_template(dummy_req)
    self.assertIsInstance(result, exception.HeatInvalidParameterValueError)
    self.m_call.assert_called_once_with(dummy_req.context, ('identify_stack', {'stack_name': stack_name}))