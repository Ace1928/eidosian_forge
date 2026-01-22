from unittest import mock
import ddt
from oslo_utils import uuidutils
import manilaclient
from manilaclient import exceptions
from manilaclient.tests.unit import utils
from manilaclient.v2 import client
def test_auth_via_token(self):
    base_url = uuidutils.generate_uuid(dashed=False)
    c = client.Client(input_auth_token='token', service_catalog_url=base_url, api_version=manilaclient.API_MAX_VERSION)
    self.assertIsNotNone(c.client)
    self.assertIsNone(c.keystone_client)