import uuid
from keystone.common import manager
from keystone.common import provider_api
from keystone.tests import unit
def test_registry_lock(self):
    provider_api.ProviderAPIs.lock_provider_registry()
    self.assertRaises(RuntimeError, self._create_manager_instance)