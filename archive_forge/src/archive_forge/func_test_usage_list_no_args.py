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
def test_usage_list_no_args(self):
    timeutils.set_time_override(datetime.datetime(2005, 2, 1, 0, 0))
    self.addCleanup(timeutils.clear_time_override)
    self.run_command('usage-list')
    self.assert_called('GET', '/os-simple-tenant-usage?' + 'start=2005-01-04T00:00:00&' + 'end=2005-02-02T00:00:00&' + 'detailed=1')