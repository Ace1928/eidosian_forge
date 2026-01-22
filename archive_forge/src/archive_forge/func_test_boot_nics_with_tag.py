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
def test_boot_nics_with_tag(self):
    cmd = 'boot --image %s --flavor 1 --nic net-id=a=c,v4-fixed-ip=10.0.0.1,tag=foo some-server' % FAKE_UUID_1
    self.run_command(cmd, api_version='2.32')
    self.assert_called_anytime('POST', '/servers', {'server': {'flavorRef': '1', 'name': 'some-server', 'imageRef': FAKE_UUID_1, 'min_count': 1, 'max_count': 1, 'networks': [{'uuid': 'a=c', 'fixed_ip': '10.0.0.1', 'tag': 'foo'}]}})