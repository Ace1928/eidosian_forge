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
def test_aggregate_update_by_id(self):
    out, err = self.run_command('aggregate-update 1 --name new_name')
    body = {'aggregate': {'name': 'new_name'}}
    self.assert_called('PUT', '/os-aggregates/1', body, pos=-2)
    self.assert_called('GET', '/os-aggregates/1', pos=-1)
    self.assertNotIn('UUID', out)