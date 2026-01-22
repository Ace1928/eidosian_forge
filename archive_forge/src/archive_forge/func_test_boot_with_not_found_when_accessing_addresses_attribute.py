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
@mock.patch.object(servers.Server, 'networks', new_callable=mock.PropertyMock)
def test_boot_with_not_found_when_accessing_addresses_attribute(self, mock_networks):
    mock_networks.side_effect = exceptions.NotFound(404, 'Instance %s could not be found.' % FAKE_UUID_1)
    ex = self.assertRaises(exceptions.CommandError, self.run_command, 'boot --flavor 1 --image %s some-server' % FAKE_UUID_2)
    self.assertIn('Instance %s could not be found.' % FAKE_UUID_1, str(ex))