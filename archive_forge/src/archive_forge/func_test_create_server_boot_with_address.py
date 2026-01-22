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
def test_create_server_boot_with_address(self):
    old_boot = self.cs.servers._boot
    access_ip_v6 = '::1'
    access_ip_v4 = '10.10.10.10'

    def wrapped_boot(url, key, *boot_args, **boot_kwargs):
        self.assertEqual(boot_kwargs['access_ip_v6'], access_ip_v6)
        self.assertEqual(boot_kwargs['access_ip_v4'], access_ip_v4)
        return old_boot(url, key, *boot_args, **boot_kwargs)
    with mock.patch.object(self.cs.servers, '_boot', wrapped_boot):
        s = self.cs.servers.create(name='My server', image=1, flavor=1, meta={'foo': 'bar'}, userdata='hello moto', key_name='fakekey', access_ip_v6=access_ip_v6, access_ip_v4=access_ip_v4, nics=self._get_server_create_default_nics())
        self.assert_request_id(s, fakes.FAKE_REQUEST_ID_LIST)
        self.assert_called('POST', '/servers')
        self.assertIsInstance(s, servers.Server)