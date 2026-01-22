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
@requests_mock.mock()
def test_upload_context(self, m):
    """Verify context (ie token) is passed to swift on upload."""
    head_req = m.head('http://127.0.0.1/glance_123', text='Some data', status_code=201)
    put_req = m.put('http://127.0.0.1/glance_123/123')
    self.config(group='swift1', swift_store_multi_tenant=True)
    store = Store(self.conf, backend='swift1')
    store.configure()
    content = b'Some data'
    pseudo_file = io.BytesIO(content)
    store.add('123', pseudo_file, len(content), context=self.ctx)
    self.assertEqual(b'0123', head_req.last_request.headers['X-Auth-Token'])
    self.assertEqual(b'0123', put_req.last_request.headers['X-Auth-Token'])