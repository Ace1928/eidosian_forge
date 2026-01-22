import json
import logging
from unittest import mock
import ddt
import fixtures
from keystoneauth1 import adapter
from keystoneauth1 import exceptions as keystone_exception
from oslo_serialization import jsonutils
from cinderclient import api_versions
import cinderclient.client
from cinderclient import exceptions
from cinderclient.tests.unit import utils
from cinderclient.tests.unit.v3 import fakes
def test_versions(self):
    v2_url = 'http://fakeurl/v2/tenants'
    v3_url = 'http://fakeurl/v3/tenants'
    unknown_url = 'http://fakeurl/v9/tenants'
    self.assertRaises(cinderclient.exceptions.UnsupportedVersion, cinderclient.client.get_volume_api_from_url, v2_url)
    self.assertEqual('3', cinderclient.client.get_volume_api_from_url(v3_url))
    self.assertRaises(cinderclient.exceptions.UnsupportedVersion, cinderclient.client.get_volume_api_from_url, unknown_url)