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
def test_microversion_with_specified_version_out_of_range(self):
    novaclient.API_MAX_VERSION = api_versions.APIVersion('2.100')
    novaclient.API_MIN_VERSION = api_versions.APIVersion('2.90')
    self.assertRaises(exceptions.CommandError, self.shell, '--os-compute-api-version 2.199 list')