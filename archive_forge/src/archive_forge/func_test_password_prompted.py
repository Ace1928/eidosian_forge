import argparse
import io
import json
import re
import sys
from unittest import mock
import ddt
import fixtures
import keystoneauth1.exceptions as ks_exc
from keystoneauth1.exceptions import DiscoveryFailure
from keystoneauth1.identity.generic.password import Password as ks_password
from keystoneauth1 import session
import requests_mock
from testtools import matchers
import cinderclient
from cinderclient import api_versions
from cinderclient.contrib import noauth
from cinderclient import exceptions
from cinderclient import shell
from cinderclient.tests.unit import fake_actions_module
from cinderclient.tests.unit.fixture_data import keystone_client
from cinderclient.tests.unit import utils
from cinderclient.tests.unit.v3 import fakes
@mock.patch('keystoneauth1.identity.v2.Password')
@mock.patch('keystoneauth1.adapter.Adapter.get_token', side_effect=ks_exc.ConnectFailure())
@mock.patch('keystoneauth1.discover.Discover', side_effect=ks_exc.ConnectFailure())
@mock.patch('sys.stdin', side_effect=mock.Mock)
@mock.patch('getpass.getpass', return_value='password')
def test_password_prompted(self, mock_getpass, mock_stdin, mock_discover, mock_token, mock_password):
    self.make_env(exclude='OS_PASSWORD')
    _shell = shell.OpenStackCinderShell()
    self.assertRaises(ks_exc.ConnectFailure, _shell.main, ['list'])
    mock_getpass.assert_called_with('OS Password: ')
    mock_password.assert_called_with(self.FAKE_ENV['OS_AUTH_URL'], password=mock_getpass.return_value, tenant_id='', tenant_name=self.FAKE_ENV['OS_PROJECT_NAME'], username=self.FAKE_ENV['OS_USERNAME'])