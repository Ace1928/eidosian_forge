from unittest import mock
from glance_store._drivers.swift import connection_manager
from glance_store._drivers.swift import store as swift_store
from glance_store import exceptions
from glance_store.tests import base
def test_single_tenant_client_cm_with_no_expiration(self):
    store = self.prepare_store()
    manager = connection_manager.SingleTenantConnectionManager(store=store, store_location=self.location, allow_reauth=True)
    store.init_client.assert_called_once_with(self.location, None)
    self.client.session.get_endpoint.assert_called_once_with(service_type=store.service_type, interface=store.endpoint_type, region_name=store.region)
    auth_ref = mock.MagicMock()
    self.client.session.auth.auth_ref = auth_ref
    auth_ref.will_expire_soon.return_value = False
    manager.get_connection()
    store.get_store_connection.assert_called_once_with('fake_token', manager.storage_url)
    self.client.session.get_auth_headers.assert_called_once_with()