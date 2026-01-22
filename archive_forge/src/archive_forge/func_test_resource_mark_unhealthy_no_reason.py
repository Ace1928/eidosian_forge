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
def test_resource_mark_unhealthy_no_reason(self):
    self.register_keystone_auth_fixture()
    stack_id = 'teststack/1'
    resource_name = 'aResource'
    self.mock_request_patch('/stacks/%s/resources/%s' % (parse.quote(stack_id), parse.quote(encodeutils.safe_encode(resource_name))), '', req_headers=False, data={'mark_unhealthy': True, 'resource_status_reason': ''})
    text = self.shell('resource-mark-unhealthy {0} {1}'.format(stack_id, resource_name))
    self.assertEqual('', text)