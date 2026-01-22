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
@mock.patch.object(store, 'get_store_from_store_identifier')
def test_get_size_with_multi_tenant_on(self, mock_get):
    """Test that single tenant uris work with multi tenant on."""
    mock_get.return_value = self.store
    uri = 'swift://%s:key@auth_address/glance/%s' % (self.swift_store_user, FAKE_UUID)
    self.config(group='swift1', swift_store_config_file=None)
    self.config(group='swift1', swift_store_multi_tenant=True)
    ctxt = mock.MagicMock()
    size = store.get_size_from_uri_and_backend(uri, 'swift1', context=ctxt)
    self.assertEqual(5120, size)