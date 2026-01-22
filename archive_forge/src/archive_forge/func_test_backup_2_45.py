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
def test_backup_2_45(self):
    """Tests the backup command with the 2.45 microversion which
        handles a different response and prints out the backup snapshot
        image details.
        """
    out, err = self.run_command('backup sample-server back1 daily 1', api_version='2.45')
    self.assertIn('back1', out)
    self.assertIn('SAVING', out)
    self.assert_called_anytime('POST', '/servers/1234/action', {'createBackup': {'name': 'back1', 'backup_type': 'daily', 'rotation': '1'}})