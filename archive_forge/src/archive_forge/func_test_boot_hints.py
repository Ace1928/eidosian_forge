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
def test_boot_hints(self):
    cmd = 'boot --image %s --flavor 1 --hint same_host=a0cf03a5-d921-4877-bb5c-86d26cf818e1 --hint same_host=8c19174f-4220-44f0-824a-cd1eeef10287 --hint query=[>=,$free_ram_mb,1024] some-server' % FAKE_UUID_1
    self.run_command(cmd)
    self.assert_called_anytime('POST', '/servers', {'server': {'flavorRef': '1', 'name': 'some-server', 'imageRef': FAKE_UUID_1, 'min_count': 1, 'max_count': 1}, 'os:scheduler_hints': {'same_host': ['a0cf03a5-d921-4877-bb5c-86d26cf818e1', '8c19174f-4220-44f0-824a-cd1eeef10287'], 'query': '[>=,$free_ram_mb,1024]'}})