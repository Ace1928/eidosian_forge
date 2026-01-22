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
def test_rebuild_preserve_ephemeral(self):
    self.run_command('rebuild sample-server %s --preserve-ephemeral' % FAKE_UUID_1)
    self.assert_called('GET', '/servers?name=sample-server', pos=0)
    self.assert_called('GET', '/servers/1234', pos=1)
    self.assert_called('GET', '/v2/images/%s' % FAKE_UUID_1, pos=2)
    self.assert_called('POST', '/servers/1234/action', {'rebuild': {'imageRef': FAKE_UUID_1, 'preserve_ephemeral': True}}, pos=3)
    self.assert_called('GET', '/flavors/1', pos=4)
    self.assert_called('GET', '/v2/images/%s' % FAKE_UUID_2, pos=5)