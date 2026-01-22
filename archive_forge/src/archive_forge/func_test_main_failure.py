import io
import re
import sys
from unittest import mock
import ddt
import fixtures
from tempest.lib.cli import output_parser
from testtools import matchers
import manilaclient
from manilaclient.common import cliutils
from manilaclient.common import constants
from manilaclient import exceptions
from manilaclient import shell
from manilaclient.tests.unit import utils
from manilaclient.tests.unit.v2 import fakes
@ddt.data({}, {'OS_AUTH_URL': 'http://foo.bar'}, {'OS_AUTH_URL': 'http://foo.bar', 'OS_USERNAME': 'foo'}, {'OS_AUTH_URL': 'http://foo.bar', 'OS_USERNAME': 'foo_user', 'OS_PASSWORD': 'foo_password'}, {'OS_TENANT_NAME': 'foo_tenant', 'OS_USERNAME': 'foo_user', 'OS_PASSWORD': 'foo_password'}, {'OS_TOKEN': 'foo_token'}, {'OS_MANILA_BYPASS_URL': 'http://foo.foo'})
def test_main_failure(self, env_vars):
    self.set_env_vars(env_vars)
    with mock.patch.object(shell, 'client') as mock_client:
        self.assertRaises(exceptions.CommandError, self.shell, 'list')
        self.assertFalse(mock_client.Client.called)