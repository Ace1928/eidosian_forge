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
@mock.patch('cinderclient.api_versions.discover_version', return_value=api_versions.APIVersion('3.0'))
@requests_mock.Mocker()
def test_noauth_plugin(self, mock_disco, mocker):
    self.assertTrue(requests_mock.mocker.Mocker, type(mocker))
    os_volume_url = 'http://example.com/volumes/v3'
    mocker.register_uri('GET', '%s/volumes/detail' % os_volume_url, text='{"volumes": []}')
    _shell = shell.OpenStackCinderShell()
    args = ['--os-endpoint', os_volume_url, '--os-auth-type', 'noauth', '--os-user-id', 'admin', '--os-project-id', 'admin', 'list']
    _shell.main(args)
    self.assertIsInstance(_shell.cs.client.session.auth, noauth.CinderNoAuthPlugin)