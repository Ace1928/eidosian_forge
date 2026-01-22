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
def test_server_tag_add_many(self):
    self.run_command('server-tag-add sample-server tag1 tag2 tag3', api_version='2.26')
    self.assert_called('PUT', '/servers/1234/tags/tag1', None, pos=-3)
    self.assert_called('PUT', '/servers/1234/tags/tag2', None, pos=-2)
    self.assert_called('PUT', '/servers/1234/tags/tag3', None, pos=-1)