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
@mock.patch.object(cinderclient.client.SessionClient, 'authenticate', side_effect=exceptions.Unauthorized('No'))
def test_session_client_debug_logger(self, mock_session):
    _shell = shell.OpenStackCinderShell()
    self.assertRaises(exceptions.CommandError, _shell.main, ['--debug', 'list'])
    self.assertEqual('keystoneauth', _shell.cs.client.logger.name)