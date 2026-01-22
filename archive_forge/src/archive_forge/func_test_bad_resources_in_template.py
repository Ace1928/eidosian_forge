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
def test_bad_resources_in_template(self):
    json_template = {'AWSTemplateFormatVersion': '2010-09-09', 'Resources': {'Type': 'AWS: : EC2: : Instance'}}
    params = {'Action': 'ValidateTemplate', 'TemplateBody': '%s' % json.dumps(json_template)}
    response = {'Error': 'Resources must contain Resource. Found a [string] instead'}
    dummy_req = self._dummy_GET_request(params)
    self._stub_enforce(dummy_req, 'ValidateTemplate')
    self.m_call.return_value = response
    response = self.controller.validate_template(dummy_req)
    expected = {'ValidateTemplateResponse': {'ValidateTemplateResult': 'Resources must contain Resource. Found a [string] instead'}}
    self.assertEqual(expected, response)
    self.m_call.assert_called_once_with(dummy_req.context, ('validate_template', {'template': json_template, 'params': None, 'files': None, 'environment_files': None, 'files_container': None, 'show_nested': False, 'ignorable_errors': None}), version='1.36')