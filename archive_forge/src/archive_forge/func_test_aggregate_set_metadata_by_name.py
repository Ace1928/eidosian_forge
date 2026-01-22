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
def test_aggregate_set_metadata_by_name(self):
    self.run_command('aggregate-set-metadata test foo=bar')
    body = {'set_metadata': {'metadata': {'foo': 'bar'}}}
    self.assert_called('POST', '/os-aggregates/1/action', body, pos=-2)
    self.assert_called('GET', '/os-aggregates/1', pos=-1)