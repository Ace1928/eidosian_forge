import argparse
import io
import re
import sys
from unittest import mock
import ddt
import fixtures
from keystoneauth1 import fixture
import requests_mock
from testtools import matchers
from novaclient import api_versions
import novaclient.client
from novaclient import exceptions
import novaclient.shell
from novaclient.tests.unit import fake_actions_module
from novaclient.tests.unit import utils
def test_microversion_with_specific_version_without_microversions(self):
    self.make_env()
    self.mock_server_version_range.return_value = (api_versions.APIVersion(), api_versions.APIVersion())
    novaclient.API_MAX_VERSION = api_versions.APIVersion('2.100')
    novaclient.API_MIN_VERSION = api_versions.APIVersion('2.1')
    self.assertRaises(exceptions.UnsupportedVersion, self.shell, '--os-compute-api-version 2.3 list')