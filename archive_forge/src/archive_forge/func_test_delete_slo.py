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
@mock.patch.object(swiftclient.client, 'delete_object')
def test_delete_slo(self, mock_del_obj):
    """
        Test we can delete an existing image stored as SLO, static large object
        """
    conf = copy.deepcopy(SWIFT_CONF)
    self.config(group='swift1', **conf)
    importlib.reload(swift)
    self.store = Store(self.conf, backend='swift1')
    self.store.configure()
    uri = 'swift://%s:key@authurl/glance/%s' % (self.swift_store_user, FAKE_UUID2)
    loc = location.get_location_from_uri_and_backend(uri, 'swift1', conf=self.conf)
    self.store.delete(loc)
    self.assertEqual(1, mock_del_obj.call_count)
    _, kwargs = mock_del_obj.call_args
    self.assertEqual('multipart-manifest=delete', kwargs.get('query_string'))