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
def test_quota_update_fixed_ip(self):
    self.run_command('quota-update 97f4c221bff44578b0300df4ef119353 --fixed-ips=5')
    self.assert_called('PUT', '/os-quota-sets/97f4c221bff44578b0300df4ef119353', {'quota_set': {'fixed_ips': 5}})