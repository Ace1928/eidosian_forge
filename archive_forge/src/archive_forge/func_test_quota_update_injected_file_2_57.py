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
def test_quota_update_injected_file_2_57(self):
    """Tests that trying to update injected_file* quota with microversion
        2.57 fails.
        """
    for quota in ('--injected-files', '--injected-file-content-bytes', '--injected-file-path-bytes'):
        cmd = 'quota-update 97f4c221bff44578b0300df4ef119353 %s=5' % quota
        self.assertRaises(SystemExit, self.run_command, cmd, api_version='2.57')