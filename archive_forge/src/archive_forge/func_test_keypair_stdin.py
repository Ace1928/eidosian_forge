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
def test_keypair_stdin(self):
    with mock.patch('sys.stdin', io.StringIO('FAKE_PUBLIC_KEY')):
        self.run_command('keypair-add --pub-key - test', api_version='2.2')
        self.assert_called('POST', '/os-keypairs', {'keypair': {'public_key': 'FAKE_PUBLIC_KEY', 'name': 'test', 'type': 'ssh'}})