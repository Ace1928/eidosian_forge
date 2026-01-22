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
def test_list_v269_with_down_cells(self):
    stdout, _stderr = self.run_command('list --minimal', api_version='2.69')
    expected = '+------+----------------+\n| ID   | Name           |\n+------+----------------+\n| 9015 |                |\n| 9014 | help           |\n| 1234 | sample-server  |\n| 5678 | sample-server2 |\n+------+----------------+\n'
    self.assertEqual(expected, stdout)
    self.assert_called('GET', '/servers')