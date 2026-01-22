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
@mock.patch('glance_store._drivers.swift.utils.is_multiple_swift_store_accounts_enabled', mock.Mock(return_value=False))
def test_multi_container_doesnt_impact_multi_tenant_add(self):
    expected_swift_size = FIVE_KB
    expected_swift_contents = b'*' * expected_swift_size
    expected_image_id = str(uuid.uuid4())
    expected_container = 'container_' + expected_image_id
    loc = 'swift+https://some_endpoint/%s/%s'
    expected_location = loc % (expected_container, expected_image_id)
    image_swift = io.BytesIO(expected_swift_contents)
    global SWIFT_PUT_OBJECT_CALLS
    SWIFT_PUT_OBJECT_CALLS = 0
    self.config(group='swift1', swift_store_container='container')
    self.config(group='swift1', swift_store_create_container_on_put=True)
    self.config(group='swift1', swift_store_multiple_containers_seed=2)
    service_catalog = [{'endpoint_links': [], 'endpoints': [{'adminURL': 'https://some_admin_endpoint', 'region': 'RegionOne', 'internalURL': 'https://some_internal_endpoint', 'publicURL': 'https://some_endpoint'}], 'type': 'object-store', 'name': 'Object Storage Service'}]
    ctxt = mock.MagicMock(user='user', tenant='tenant', auth_token='123', service_catalog=service_catalog)
    store = swift.MultiTenantStore(self.conf, backend='swift1')
    store.configure()
    location, size, checksum, metadata = store.add(expected_image_id, image_swift, expected_swift_size, context=ctxt)
    self.assertEqual('swift1', metadata['store'])
    self.assertEqual(expected_location, location)