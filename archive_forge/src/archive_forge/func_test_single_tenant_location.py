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
def test_single_tenant_location(self):
    conf = copy.deepcopy(SWIFT_CONF)
    conf['swift_store_container'] = 'container'
    conf_file = 'glance-swift.conf'
    self.swift_config_file = self.copy_data_file(conf_file, self.test_dir)
    conf.update({'swift_store_config_file': self.swift_config_file})
    conf['default_swift_reference'] = 'ref1'
    self.config(group='swift1', **conf)
    importlib.reload(swift)
    store = swift.SingleTenantStore(self.conf, backend='swift1')
    store.configure()
    location = store.create_location('image-id')
    self.assertEqual('swift+https', location.scheme)
    self.assertEqual('https://example.com', location.swift_url)
    self.assertEqual('container', location.container)
    self.assertEqual('image-id', location.obj)
    self.assertEqual('tenant:user1', location.user)
    self.assertEqual('key1', location.key)