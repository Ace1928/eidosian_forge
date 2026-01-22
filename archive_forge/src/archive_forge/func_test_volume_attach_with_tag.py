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
def test_volume_attach_with_tag(self):
    out = self.run_command('volume-attach --tag test_tag sample-server Work /dev/vdb', api_version='2.49')[0]
    self.assert_called('POST', '/servers/1234/os-volume_attachments', {'volumeAttachment': {'device': '/dev/vdb', 'volumeId': 'Work', 'tag': 'test_tag'}})
    self.assertNotIn('test-tag', out)