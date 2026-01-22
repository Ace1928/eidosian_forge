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
def test_boot_invalid_nics_v2_37(self):
    """This is a negative test to make sure we fail with the correct
        message.
        """
    cmd = 'boot --image %s --flavor 1 --nic net-id=1 --nic auto some-server' % FAKE_UUID_1
    ex = self.assertRaises(exceptions.CommandError, self.run_command, cmd, api_version='2.37')
    self.assertIn('auto,none', str(ex))