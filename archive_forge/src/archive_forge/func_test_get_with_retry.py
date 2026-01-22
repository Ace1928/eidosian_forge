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
def test_get_with_retry(self):
    """
        Test a retrieval where Swift does not get the full image in a single
        request.
        """
    uri = 'swift://%s:key@auth_address/glance/%s' % (self.swift_store_user, FAKE_UUID)
    loc = location.get_location_from_uri_and_backend(uri, 'swift1', conf=self.conf)
    ctxt = mock.MagicMock()
    image_swift, image_size = self.store.get(loc, context=ctxt)
    resp_full = b''.join([chunk for chunk in image_swift.wrapped])
    resp_half = resp_full[:len(resp_full) // 2]
    resp_half = io.BytesIO(resp_half)
    manager = self.store.get_manager(loc.store_location, ctxt)
    image_swift.wrapped = swift.swift_retry_iter(resp_half, image_size, self.store, loc.store_location, manager)
    self.assertEqual(5120, image_size)
    expected_data = b'*' * FIVE_KB
    data = b''
    for chunk in image_swift:
        data += chunk
    self.assertEqual(expected_data, data)