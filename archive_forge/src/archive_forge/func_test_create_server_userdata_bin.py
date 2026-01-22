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
def test_create_server_userdata_bin(self):
    kwargs = {}
    if self.supports_files:
        kwargs['files'] = {'/etc/passwd': 'some data', '/tmp/foo.txt': io.StringIO('data')}
    with tempfile.TemporaryFile(mode='wb+') as bin_file:
        original_data = os.urandom(1024)
        bin_file.write(original_data)
        bin_file.flush()
        bin_file.seek(0)
        s = self.cs.servers.create(name='My server', image=1, flavor=1, meta={'foo': 'bar'}, userdata=bin_file, key_name='fakekey', nics=self._get_server_create_default_nics(), **kwargs)
        self.assert_request_id(s, fakes.FAKE_REQUEST_ID_LIST)
        self.assert_called('POST', '/servers')
        self.assertIsInstance(s, servers.Server)
        body = self.requests_mock.last_request.json()
        transferred_data = body['server']['user_data']
        transferred_data = base64.b64decode(transferred_data)
        self.assertEqual(original_data, transferred_data)