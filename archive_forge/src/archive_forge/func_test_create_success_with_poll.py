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
def test_create_success_with_poll(self):
    self.register_keystone_auth_fixture()
    stack_create_resp_dict = {'stack': {'id': 'teststack2/2', 'stack_name': 'teststack2', 'stack_status': 'CREATE_IN_PROGRESS', 'creation_time': '2012-10-25T01:58:47Z'}}
    self.mock_request_post('/stacks', stack_create_resp_dict, data=mock.ANY, req_headers=True, status_code=201)
    self.mock_stack_list()
    stack_show_resp_dict = {'stack': {'id': '1', 'stack_name': 'teststack', 'stack_status': 'CREATE_COMPLETE', 'creation_time': '2012-10-25T01:58:47Z'}}
    event_list_resp_dict = self.event_list_resp_dict(stack_name='teststack2')
    stack_id = 'teststack2'
    self.mock_request_get('/stacks/teststack2', stack_show_resp_dict)
    self.mock_request_get('/stacks/%s/events?sort_dir=asc' % stack_id, event_list_resp_dict)
    self.mock_request_get('/stacks/teststack2', stack_show_resp_dict)
    template_file = os.path.join(TEST_VAR_DIR, 'minimal.template')
    create_text = self.shell('stack-create teststack2 --poll 4 --template-file=%s --parameters="InstanceType=m1.large;DBUsername=wp;DBPassword=verybadpassword;KeyName=heat_key;LinuxDistribution=F17"' % template_file)
    required = ['id', 'stack_name', 'stack_status', '2', 'teststack2', 'IN_PROGRESS', '14:14:30', '2013-12-05', 'CREATE_IN_PROGRESS', 'state changed', '14:14:31', 'testresource', '14:14:32', 'CREATE_COMPLETE', '14:14:33']
    for r in required:
        self.assertRegex(create_text, r)