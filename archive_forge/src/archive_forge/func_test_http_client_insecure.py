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
@mock.patch.object(cinderclient.client.HTTPClient, 'authenticate', side_effect=exceptions.Unauthorized('No'))
@mock.patch.object(cinderclient.shell.OpenStackCinderShell, '_get_keystone_session', return_value=None)
def test_http_client_insecure(self, mock_authenticate, mock_session):
    self.make_env(include={'CINDERCLIENT_INSECURE': True})
    _shell = shell.OpenStackCinderShell()
    self.assertRaises(exceptions.CommandError, _shell.main, ['list'])
    self.assertEqual(False, _shell.cs.client.verify_cert)