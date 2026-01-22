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
def test_volume_attach_without_delete_on_termination(self):
    self.run_command('volume-attach sample-server Work', api_version='2.79')
    self.assert_called('POST', '/servers/1234/os-volume_attachments', {'volumeAttachment': {'volumeId': 'Work'}})