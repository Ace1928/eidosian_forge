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
def test_stack_update_with_existing_template(self):
    self.register_keystone_auth_fixture()
    expected_data = {'files': {}, 'environment': {}, 'template': None, 'parameters': {}}
    self.mock_request_patch('/stacks/teststack2/2', 'The request is accepted for processing.', data=expected_data)
    self.mock_stack_list()
    update_text = self.shell('stack-update teststack2/2 --existing')
    required = ['stack_name', 'id', 'teststack2', '1']
    for r in required:
        self.assertRegex(update_text, r)