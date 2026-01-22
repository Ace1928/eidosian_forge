from unittest import mock
from glance_store._drivers.swift import connection_manager
from glance_store._drivers.swift import store as swift_store
from glance_store import exceptions
from glance_store.tests import base
def prepare_store(self, multi_tenant=False):
    if multi_tenant:
        store = mock.create_autospec(swift_store.MultiTenantStore, conf=self.conf)
    else:
        store = mock.create_autospec(swift_store.SingleTenantStore, service_type='swift', endpoint_type='internal', region=None, conf=self.conf, auth_version='3')
    store.backend_group = None
    store.conf_endpoint = None
    store.init_client.return_value = self.client
    return store