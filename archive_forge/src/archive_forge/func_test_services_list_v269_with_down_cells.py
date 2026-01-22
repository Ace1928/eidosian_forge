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
def test_services_list_v269_with_down_cells(self):
    """Tests nova service-list at the 2.69 microversion."""
    stdout, _stderr = self.run_command('service-list', api_version='2.69')
    self.assertEqual('+--------------------------------------+--------------+-----------+------+----------+-------+---------------------+-----------------+-------------+\n| Id                                   | Binary       | Host      | Zone | Status   | State | Updated_at          | Disabled Reason | Forced down |\n+--------------------------------------+--------------+-----------+------+----------+-------+---------------------+-----------------+-------------+\n| 75e9eabc-ed3b-4f11-8bba-add1e7e7e2de | nova-compute | host1     | nova | enabled  | up    | 2012-10-29 13:42:02 |                 |             |\n| 1f140183-c914-4ddf-8757-6df73028aa86 | nova-compute | host1     | nova | disabled | down  | 2012-09-18 08:03:38 |                 |             |\n|                                      | nova-compute | host-down |      | UNKNOWN  |       |                     |                 |             |\n+--------------------------------------+--------------+-----------+------+----------+-------+---------------------+-----------------+-------------+\n', stdout)
    self.assert_called('GET', '/os-services')