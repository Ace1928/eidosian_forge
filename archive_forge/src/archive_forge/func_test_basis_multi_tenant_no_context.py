from unittest import mock
from glance_store._drivers.swift import connection_manager
from glance_store._drivers.swift import store as swift_store
from glance_store import exceptions
from glance_store.tests import base
def test_basis_multi_tenant_no_context(self):
    store = self.prepare_store(multi_tenant=True)
    self.assertRaises(exceptions.BadStoreConfiguration, connection_manager.MultiTenantConnectionManager, store=store, store_location=self.location)