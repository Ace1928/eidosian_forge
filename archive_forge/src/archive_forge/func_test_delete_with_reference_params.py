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
def test_delete_with_reference_params(self):
    """
        Test we can delete an existing image in the swift store
        """
    conf = copy.deepcopy(SWIFT_CONF)
    self.config(group='swift1', **conf)
    importlib.reload(swift)
    self.mock_keystone_client()
    self.store = Store(self.conf, backend='swift1')
    self.store.configure()
    uri = 'swift+config://ref1/glance/%s' % FAKE_UUID
    loc = location.get_location_from_uri_and_backend(uri, 'swift1', conf=self.conf)
    self.store.delete(loc)
    self.assertRaises(exceptions.NotFound, self.store.get, loc)