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
def test_create_server_with_tags(self):
    self.cs.servers.create(name='My server', image=1, flavor=1, meta={'foo': 'bar'}, userdata='hello moto', key_name='fakekey', nics=self._get_server_create_default_nics(), tags=['tag1', 'tag2'])
    self.assert_called('POST', '/servers', {'server': {'flavorRef': '1', 'imageRef': '1', 'key_name': 'fakekey', 'max_count': 1, 'metadata': {'foo': 'bar'}, 'min_count': 1, 'name': 'My server', 'networks': 'auto', 'tags': ['tag1', 'tag2'], 'user_data': 'aGVsbG8gbW90bw=='}})