import base64
import io
import os
import tempfile
from unittest import mock
from novaclient import api_versions
from novaclient import exceptions
from novaclient.tests.unit.fixture_data import client
from novaclient.tests.unit.fixture_data import floatingips
from novaclient.tests.unit.fixture_data import servers as data
from novaclient.tests.unit import utils
from novaclient.tests.unit.v2 import fakes
from novaclient.v2 import servers
def wrapped_boot(url, key, *boot_args, **boot_kwargs):
    self.assertEqual(boot_kwargs['access_ip_v6'], access_ip_v6)
    self.assertEqual(boot_kwargs['access_ip_v4'], access_ip_v4)
    return old_boot(url, key, *boot_args, **boot_kwargs)