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
@mock.patch('novaclient.v2.shell._find_server')
@mock.patch('os.system')
def test_ssh_multinet(self, mock_system, mock_find_server):

    class FakeResources(object):
        addresses = {'skynet': [{'version': 4, 'addr': '1.1.1.1', 'OS-EXT-IPS:type': 'fixed'}, {'version': 4, 'addr': '2.2.2.2'}, {'version': 6, 'addr': '2607:f0d0:1002::4', 'OS-EXT-IPS:type': 'fixed'}], 'other': [{'version': 4, 'addr': '2.3.4.5'}, {'version': 6, 'addr': '7612:a1b2:2004::6'}]}
    mock_find_server.return_value = FakeResources()
    self.run_command('ssh --network other server')
    mock_system.assert_called_with('ssh -4 -p22  root@2.3.4.5 ')
    self.run_command('ssh --ipv6 --network other server')
    mock_system.assert_called_with('ssh -6 -p22  root@7612:a1b2:2004::6 ')
    self.assertRaises(exceptions.ResourceNotFound, self.run_command, 'ssh --ipv6 --network nonexistent server')