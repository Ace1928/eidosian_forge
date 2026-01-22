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
def test_stack_update_with_patched_and_default_parameters(self):
    self.register_keystone_auth_fixture()
    template_file = os.path.join(TEST_VAR_DIR, 'minimal.template')
    template_data = open(template_file).read()
    expected_data = {'files': {}, 'environment': {}, 'template': jsonutils.loads(template_data), 'parameters': {'"KeyPairName': 'updated_key"'}, 'clear_parameters': ['InstanceType', 'DBUsername', 'DBPassword', 'KeyPairName', 'LinuxDistribution'], 'disable_rollback': False}
    self.mock_request_patch('/stacks/teststack2/2', 'The request is accepted for processing.', data=expected_data)
    self.mock_stack_list()
    update_text = self.shell('stack-update teststack2/2 --template-file=%s --enable-rollback --existing --parameters="KeyPairName=updated_key" --clear-parameter=InstanceType --clear-parameter=DBUsername --clear-parameter=DBPassword --clear-parameter=KeyPairName --clear-parameter=LinuxDistribution' % template_file)
    required = ['stack_name', 'id', 'teststack2', '1']
    for r in required:
        self.assertRegex(update_text, r)