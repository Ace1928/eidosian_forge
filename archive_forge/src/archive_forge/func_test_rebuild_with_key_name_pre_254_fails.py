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
def test_rebuild_with_key_name_pre_254_fails(self):
    self.cs.api_version = api_versions.APIVersion('2.53')
    ex = self.assertRaises(exceptions.UnsupportedAttribute, self.cs.servers.rebuild, '1234', fakes.FAKE_IMAGE_UUID_1, key_name='test_keypair')
    self.assertIn('key_name', str(ex.message))