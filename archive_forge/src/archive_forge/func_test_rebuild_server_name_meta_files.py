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
def test_rebuild_server_name_meta_files(self):
    files = {'/etc/passwd': 'some data'}
    s = self.cs.servers.get(1234)
    ex = self.assertRaises(exceptions.UnsupportedAttribute, s.rebuild, image=1, name='new', meta={'foo': 'bar'}, files=files)
    self.assertIn('files', str(ex))