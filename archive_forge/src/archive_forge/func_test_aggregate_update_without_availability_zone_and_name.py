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
def test_aggregate_update_without_availability_zone_and_name(self):
    ex = self.assertRaises(exceptions.CommandError, self.run_command, 'aggregate-update test')
    self.assertIn("Either '--name <name>' or '--availability-zone <availability-zone>' must be specified.", str(ex))