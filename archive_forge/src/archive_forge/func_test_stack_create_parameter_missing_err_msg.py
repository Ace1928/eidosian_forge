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
def test_stack_create_parameter_missing_err_msg(self):
    self.register_keystone_auth_fixture()
    resp_dict = {'error': {'message': 'The Parameter (key_name) was not provided.', 'type': 'UserParameterMissing'}}
    self.requests.post('http://heat.example.com/stacks', status_code=400, headers={'Content-Type': 'application/json'}, json=resp_dict)
    template_file = os.path.join(TEST_VAR_DIR, 'minimal.template')
    self.shell_error('stack-create -f %s stack' % template_file, 'The Parameter \\(key_name\\) was not provided.', exception=exc.HTTPBadRequest)