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
def test_describe_stack_resources_physical(self):
    stack_name = 'wordpress'
    identity = dict(identifier.HeatIdentifier('t', stack_name, '6'))
    params = {'Action': 'DescribeStackResources', 'LogicalResourceId': 'WikiDatabase', 'PhysicalResourceId': 'a3455d8c-9f88-404d-a85b-5315293e67de'}
    dummy_req = self._dummy_GET_request(params)
    self._stub_enforce(dummy_req, 'DescribeStackResources')
    engine_resp = [{u'description': u'', u'resource_identity': {u'tenant': u't', u'stack_name': u'wordpress', u'stack_id': u'6', u'path': u'resources/WikiDatabase'}, u'stack_name': u'wordpress', u'resource_name': u'WikiDatabase', u'resource_status_reason': None, u'updated_time': u'2012-07-23T13:06:00Z', u'stack_identity': {u'tenant': u't', u'stack_name': u'wordpress', u'stack_id': u'6', u'path': u''}, u'resource_action': u'CREATE', u'resource_status': u'COMPLETE', u'physical_resource_id': u'a3455d8c-9f88-404d-a85b-5315293e67de', u'resource_type': u'AWS::EC2::Instance', u'metadata': {u'ensureRunning': u'truetrue'}}]
    self.m_call.side_effect = [identity, engine_resp]
    args = {'stack_identity': identity, 'resource_name': dummy_req.params.get('LogicalResourceId')}
    response = self.controller.describe_stack_resources(dummy_req)
    expected = {'DescribeStackResourcesResponse': {'DescribeStackResourcesResult': {'StackResources': [{'StackId': u'arn:openstack:heat::t:stacks/wordpress/6', 'ResourceStatus': u'CREATE_COMPLETE', 'Description': u'', 'ResourceType': u'AWS::EC2::Instance', 'Timestamp': u'2012-07-23T13:06:00Z', 'ResourceStatusReason': None, 'StackName': u'wordpress', 'PhysicalResourceId': u'a3455d8c-9f88-404d-a85b-5315293e67de', 'LogicalResourceId': u'WikiDatabase'}]}}}
    self.assertEqual(expected, response)
    self.assertEqual([mock.call(dummy_req.context, ('find_physical_resource', {'physical_resource_id': 'a3455d8c-9f88-404d-a85b-5315293e67de'})), mock.call(dummy_req.context, ('describe_stack_resources', args))], self.m_call.call_args_list)