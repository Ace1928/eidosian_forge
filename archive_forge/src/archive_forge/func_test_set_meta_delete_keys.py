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
def test_set_meta_delete_keys(self):
    self.run_command('meta 1234 delete key1 key2')
    self.assert_called('DELETE', '/servers/1234/metadata/key1')
    self.assert_called('DELETE', '/servers/1234/metadata/key2', pos=-2)