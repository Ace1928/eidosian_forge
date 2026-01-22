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
def test_hypervisor_stats_v2_88(self):
    """Tests nova hypervisor-stats at the 2.88 microversion."""
    ex = self.assertRaises(exceptions.CommandError, self.run_command, 'hypervisor-stats', api_version='2.88')
    self.assertIn('The hypervisor-stats command is not supported in API version 2.88 or later.', str(ex))