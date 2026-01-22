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
def test_boot_invalid_min_count(self):
    cmd = 'boot --image %s --flavor 1 --min-count 0  server' % FAKE_UUID_1
    self.assertRaises(exceptions.CommandError, self.run_command, cmd)