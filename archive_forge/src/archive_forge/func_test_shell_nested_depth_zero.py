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
def test_shell_nested_depth_zero(self):
    self.register_keystone_auth_fixture()
    resp_dict = {'events': [{'id': 'eventid1'}, {'id': 'eventid2'}]}
    stack_id = 'teststack/1'
    self.mock_request_get('/stacks/%s/events?sort_dir=asc' % stack_id, resp_dict)
    list_text = self.shell('event-list %s --nested-depth 0' % stack_id)
    required = ['id', 'eventid1', 'eventid2']
    for r in required:
        self.assertRegex(list_text, r)