from unittest import mock
from glance_store._drivers.swift import connection_manager
from glance_store._drivers.swift import store as swift_store
from glance_store import exceptions
from glance_store.tests import base
def test_basic_multi_tenant_cm_init(self):
    store = self.prepare_store(multi_tenant=True)
    manager = connection_manager.MultiTenantConnectionManager(store=store, store_location=self.location, context=self.context)
    store.get_store_connection.assert_called_once_with(self.context.auth_token, manager.storage_url)