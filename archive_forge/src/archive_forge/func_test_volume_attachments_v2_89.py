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
def test_volume_attachments_v2_89(self):
    out = self.run_command('volume-attachments 1234', api_version='2.89')[0]
    self.assert_called('GET', '/servers/1234/os-volume_attachments')
    self.assertNotIn('| ID', out)
    self.assertIn('ATTACHMENT ID', out)
    self.assertIn('BDM UUID', out)