import argparse
import base64
import builtins
import collections
import datetime
import io
import os
from unittest import mock
import fixtures
from oslo_utils import timeutils
import testtools
import novaclient
from novaclient import api_versions
from novaclient import base
import novaclient.client
from novaclient import exceptions
import novaclient.shell
from novaclient.tests.unit import utils
from novaclient.tests.unit.v2 import fakes
from novaclient.v2 import servers
import novaclient.v2.shell
def test_service_force_down_v2_53(self):
    """Tests nova service-force-down at the 2.53 microversion."""
    self.run_command('service-force-down %s' % fakes.FAKE_SERVICE_UUID_1, api_version='2.53')
    body = {'forced_down': True}
    self.assert_called('PUT', '/os-services/%s' % fakes.FAKE_SERVICE_UUID_1, body)