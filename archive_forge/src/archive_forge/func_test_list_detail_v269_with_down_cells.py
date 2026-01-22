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
def test_list_detail_v269_with_down_cells(self):
    """Tests nova list at the 2.69 microversion."""
    stdout, _stderr = self.run_command('list', api_version='2.69')
    self.assertIn('+------+----------------+---------+------------+-------------+----------------------------------------------+\n| ID   | Name           | Status  | Task State | Power State | Networks                                     |\n+------+----------------+---------+------------+-------------+----------------------------------------------+\n| 9015 |                | UNKNOWN | N/A        | N/A         |                                              |\n| 9014 | help           | ACTIVE  | N/A        | N/A         |                                              |\n| 1234 | sample-server  | BUILD   | N/A        | N/A         | private=10.11.12.13; public=1.2.3.4, 5.6.7.8 |\n| 5678 | sample-server2 | ACTIVE  | N/A        | N/A         | private=10.13.12.13; public=4.5.6.7, 5.6.9.8 |\n| 9012 | sample-server3 | ACTIVE  | N/A        | N/A         | private=10.13.12.13; public=4.5.6.7, 5.6.9.8 |\n| 9013 | sample-server4 | ACTIVE  | N/A        | N/A         |                                              |\n+------+----------------+---------+------------+-------------+----------------------------------------------+\n', stdout)
    self.assert_called('GET', '/servers/detail')