import uuid
from keystone.common import manager
from keystone.common import provider_api
from keystone.tests import unit
def test_registry_duplicate(self):
    test_manager = self._create_manager_instance()
    self.assertRaises(provider_api.DuplicateProviderError, self._create_manager_instance, provides_api=test_manager._provides_api)