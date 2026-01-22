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
@mock.patch.object(self.cs.servers, '_boot', wrapped_boot)
def test_create_server_from_volume():
    s = self.cs.servers.create(name='My server', image=1, flavor=1, meta={'foo': 'bar'}, userdata='hello moto', key_name='fakekey', block_device_mapping=bdm, nics=nics)
    self.assert_request_id(s, fakes.FAKE_REQUEST_ID_LIST)
    self.assert_called('POST', '/servers')
    self.assertIsInstance(s, servers.Server)