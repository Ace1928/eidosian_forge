import os
import sys
import tempfile
from unittest import mock
import uuid
import fixtures
import io
from keystoneauth1 import fixture as keystone_fixture
from oslo_serialization import jsonutils
from oslo_utils import encodeutils
from requests_mock.contrib import fixture as rm_fixture
import testscenarios
import testtools
from urllib import parse
from urllib import request
import yaml
from heatclient._i18n import _
from heatclient.common import http
from heatclient.common import utils
from heatclient import exc
import heatclient.shell
from heatclient.tests.unit import fakes
import heatclient.v1.shell
def test_deploy_list(self):
    self.register_keystone_auth_fixture()
    resp_dict = {'software_deployments': [{'status': 'COMPLETE', 'server_id': '123', 'config_id': '18c4fc03-f897-4a1d-aaad-2b7622e60257', 'output_values': {'deploy_stdout': '', 'deploy_stderr': '', 'deploy_status_code': 0, 'result': 'The result value'}, 'input_values': {}, 'action': 'CREATE', 'status_reason': 'Outputs received', 'id': 'defg'}]}
    self.mock_request_get('/software_deployments?', resp_dict)
    self.mock_request_get('/software_deployments?server_id=123', resp_dict)
    list_text = self.shell('deployment-list')
    required = ['id', 'config_id', 'server_id', 'action', 'status', 'creation_time', 'status_reason']
    for r in required:
        self.assertRegex(list_text, r)
    self.assertNotRegex(list_text, 'parent')
    list_text = self.shell('deployment-list -s 123')
    for r in required:
        self.assertRegex(list_text, r)
    self.assertNotRegex(list_text, 'parent')