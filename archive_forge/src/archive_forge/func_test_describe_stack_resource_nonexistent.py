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
def test_describe_stack_resource_nonexistent(self):
    stack_name = 'wordpress'
    identity = dict(identifier.HeatIdentifier('t', stack_name, '6'))
    params = {'Action': 'DescribeStackResource', 'StackName': stack_name, 'LogicalResourceId': 'wibble'}
    dummy_req = self._dummy_GET_request(params)
    self._stub_enforce(dummy_req, 'DescribeStackResource')
    exc = heat_exception.ResourceNotFound(resource_name='test', stack_name='test')
    self.m_call.side_effect = [identity, exc]
    args = {'stack_identity': identity, 'resource_name': dummy_req.params.get('LogicalResourceId'), 'with_attr': False}
    result = self.controller.describe_stack_resource(dummy_req)
    self.assertIsInstance(result, exception.HeatInvalidParameterValueError)
    self.assertEqual([mock.call(dummy_req.context, ('identify_stack', {'stack_name': stack_name})), mock.call(dummy_req.context, ('describe_stack_resource', args), version='1.2')], self.m_call.call_args_list)