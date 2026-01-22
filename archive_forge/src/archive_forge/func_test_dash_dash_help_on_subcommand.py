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
@ddt.data('backup-create --help', '--help backup-create')
def test_dash_dash_help_on_subcommand(self, cmd):
    required = ['.*?^Creates a volume backup.']
    help_text = self.shell(cmd)
    for r in required:
        self.assertThat(help_text, matchers.MatchesRegex(r, re.DOTALL | re.MULTILINE))