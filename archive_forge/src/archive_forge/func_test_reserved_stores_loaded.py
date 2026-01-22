from unittest import mock
from oslo_config import cfg
import glance_store as store
from glance_store import backend
from glance_store import location
from glance_store import multi_backend
from glance_store.tests import base
def test_reserved_stores_loaded(self):
    store = multi_backend.get_store_from_store_identifier('consuming_service_reserved_store')
    self.assertIsNotNone(store)
    self.assertEqual(self.reserved_stores, multi_backend._RESERVED_STORES)
    self.assertEqual('consuming_service_reserved_store', store.backend_group)