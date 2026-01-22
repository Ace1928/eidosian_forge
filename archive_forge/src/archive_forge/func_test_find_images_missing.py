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
def test_find_images_missing(self):
    """Test find_images() where one of the images is not found."""
    self.assertRaises(exceptions.CommandError, novaclient.v2.shell._find_images, self.shell.cs, [FAKE_UUID_1, 'foo'])