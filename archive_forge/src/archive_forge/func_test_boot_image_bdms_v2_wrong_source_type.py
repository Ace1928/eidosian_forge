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
def test_boot_image_bdms_v2_wrong_source_type(self):
    self.assertRaises(exceptions.CommandError, self.run_command, 'boot --flavor 1 --image %s --block-device id=fake-id,source=fake,device=vda,size=1,format=ext4,type=disk,shutdown=preserve some-server' % FAKE_UUID_1)