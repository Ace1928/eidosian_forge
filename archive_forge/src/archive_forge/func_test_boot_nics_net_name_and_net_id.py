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
@mock.patch('novaclient.v2.shell._find_network_id', return_value='net-id')
def test_boot_nics_net_name_and_net_id(self, mock_find_network_id):
    cmd = 'boot --image %s --flavor 1 --nic net-name=some-net,net-id=some-id some-server' % FAKE_UUID_1
    self.assertRaises(exceptions.CommandError, self.run_command, cmd)