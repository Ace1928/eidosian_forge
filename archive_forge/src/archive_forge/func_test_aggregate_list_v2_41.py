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
def test_aggregate_list_v2_41(self):
    out, err = self.run_command('aggregate-list', api_version='2.41')
    self.assert_called('GET', '/os-aggregates')
    self.assertIn('UUID', out)
    self.assertIn('80785864-087b-45a5-a433-b20eac9b58aa', out)
    self.assertIn('30827713-5957-4b68-8fd3-ccaddb568c24', out)
    self.assertIn('9a651b22-ce3f-4a87-acd7-98446ef591c4', out)