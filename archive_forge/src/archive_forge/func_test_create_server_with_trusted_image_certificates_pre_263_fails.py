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
def test_create_server_with_trusted_image_certificates_pre_263_fails(self):
    self.cs.api_version = api_versions.APIVersion('2.62')
    ex = self.assertRaises(exceptions.UnsupportedAttribute, self.cs.servers.create, name='My server', image=1, flavor=1, meta={'foo': 'bar'}, userdata='hello moto', key_name='fakekey', nics=self._get_server_create_default_nics(), trusted_image_certificates=['id1', 'id2'])
    self.assertIn('trusted_image_certificates', str(ex))