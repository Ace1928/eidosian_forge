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
def test_events_list_err_rpcerr(self):
    stack_name = 'wordpress'
    identity = dict(identifier.HeatIdentifier('t', stack_name, '6'))
    params = {'Action': 'DescribeStackEvents', 'StackName': stack_name}
    dummy_req = self._dummy_GET_request(params)
    self._stub_enforce(dummy_req, 'DescribeStackEvents')

    class FakeExc(Exception):
        pass
    self.m_call.side_effect = [identity, FakeExc]
    result = self.controller.events_list(dummy_req)
    self.assertIsInstance(result, exception.HeatInternalFailureError)
    self.assertEqual([mock.call(dummy_req.context, ('identify_stack', {'stack_name': stack_name})), mock.call(dummy_req.context, mock.ANY, version='1.31')], self.m_call.call_args_list)