import copy
from unittest import mock
import fixtures
import hashlib
import http.client
import importlib
import io
import tempfile
import uuid
from oslo_config import cfg
from oslo_utils import encodeutils
from oslo_utils.secretutils import md5
from oslo_utils import units
import requests_mock
import swiftclient
from glance_store._drivers.swift import connection_manager as manager
from glance_store._drivers.swift import store as swift
from glance_store._drivers.swift import utils as sutils
from glance_store import capabilities
from glance_store import exceptions
from glance_store import location
import glance_store.multi_backend as store
from glance_store.tests import base
from glance_store.tests.unit import test_store_capabilities
def test_get_with_http_auth(self):
    """
        Test a retrieval from Swift with an HTTP authurl. This is
        specified either via a Location header with swift+http:// or using
        http:// in the swift_store_auth_address config value
        """
    loc = location.get_location_from_uri_and_backend('swift+http://%s:key@auth_address/glance/%s' % (self.swift_store_user, FAKE_UUID), 'swift1', conf=self.conf)
    ctxt = mock.MagicMock()
    image_swift, image_size = self.store.get(loc, context=ctxt)
    self.assertEqual(5120, image_size)
    expected_data = b'*' * FIVE_KB
    data = b''
    for chunk in image_swift:
        data += chunk
    self.assertEqual(expected_data, data)